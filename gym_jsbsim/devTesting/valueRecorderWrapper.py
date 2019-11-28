import numpy as np
from gym import Wrapper

class ValueRecorderWrapper(Wrapper):
    def __init__(self, env, ):
        super(ValueRecorderWrapper, self).__init__(env)
        ...


    def step(self, action):
        self._before_step(action)
        observation, reward, done, info = self.env.step(action)
        done = self._after_step(observation, reward, done, info)
        #add aciton, observation, reward, done, info to store

        return observation, reward, done, info

    def reset(self, **kwargs):
        self._before_reset()
        observation = self.env.reset(**kwargs)
        self._after_reset(observation)
        #add aciton, observation, reward, done, info to store

        return observation

    def _start(self, directory, video_callable=None, force=False, resume=False,
              write_upon_reset=False, uid=None, mode=None):
        """Start monitoring.
        Args:
            directory (str): A per-training run directory where to record stats.
            video_callable (Optional[function, False]): function that takes in the index of the episode and outputs a boolean, indicating whether we should record a video on this episode. The default (for video_callable is None) is to take perfect cubes, capped at 1000. False disables video recording.
            force (bool): Clear out existing training data from this directory (by deleting every file prefixed with "openaigym.").
            resume (bool): Retain the training data already in this directory, which will be merged with our new data
            write_upon_reset (bool): Write the manifest file on each reset. (This is currently a JSON file, so writing it is somewhat expensive.)
            uid (Optional[str]): A unique id used as part of the suffix for the file. By default, uses os.getpid().
            mode (['evaluation', 'training']): Whether this is an evaluation or training episode.
        """
        if self.env.spec is None:
            logger.warn("Trying to monitor an environment which has no 'spec' set. This usually means you did not create it via 'gym.make', and is recommended only for advanced users.")
            env_id = '(unknown)'
        else:
            env_id = self.env.spec.id

        if not os.path.exists(directory):
            logger.info('Creating monitor directory %s', directory)
            if six.PY3:
                os.makedirs(directory, exist_ok=True)
            else:
                os.makedirs(directory)

        if video_callable is None:
            video_callable = capped_cubic_video_schedule
        elif video_callable == False:
            video_callable = disable_videos
        elif not callable(video_callable):
            raise error.Error('You must provide a function, None, or False for video_callable, not {}: {}'.format(type(video_callable), video_callable))
        self.video_callable = video_callable

        # Check on whether we need to clear anything
        if force:
            clear_monitor_files(directory)
        elif not resume:
            training_manifests = detect_training_manifests(directory)
            if len(training_manifests) > 0:
                raise error.Error('''Trying to write to monitor directory {} with existing monitor files: {}.
 You should use a unique directory for each training run, or use 'force=True' to automatically clear previous monitor files.'''.format(directory, ', '.join(training_manifests[:5])))

        self._monitor_id = monitor_closer.register(self)

        self.enabled = True
        self.directory = os.path.abspath(directory)
        # We use the 'openai-gym' prefix to determine if a file is
        # ours
        self.file_prefix = FILE_PREFIX
        self.file_infix = '{}.{}'.format(self._monitor_id, uid if uid else os.getpid())

        self.stats_recorder = stats_recorder.StatsRecorder(directory, '{}.episode_batch.{}'.format(self.file_prefix, self.file_infix), autoreset=self.env_semantics_autoreset, env_id=env_id)

        if not os.path.exists(directory): os.mkdir(directory)
        self.write_upon_reset = write_upon_reset

        if mode is not None:
            self._set_mode(mode)

    def _flush(self, force=False):
        """Flush all relevant monitor information to disk."""
        if not self.write_upon_reset and not force:
            return

        self.stats_recorder.flush()

        # Give it a very distiguished name, since we need to pick it
        # up from the filesystem later.
        path = os.path.join(self.directory, '{}.manifest.{}.manifest.json'.format(self.file_prefix, self.file_infix))
        logger.debug('Writing training manifest file to %s', path)
        with atomic_write.atomic_write(path) as f:
            # We need to write relative paths here since people may
            # move the training_dir around. It would be cleaner to
            # already have the basenames rather than basename'ing
            # manually, but this works for now.
            json.dump({
                'stats': os.path.basename(self.stats_recorder.path),
                'videos': [(os.path.basename(v), os.path.basename(m))
                           for v, m in self.videos],
                'env_info': self._env_info(),
            }, f, default=json_encode_np)

    def close(self):
        """Flush all monitor data to disk and close any open rending windows."""
        super(Monitor, self).close()

        if not self.enabled:
            return
        self.stats_recorder.close()
        self._flush(force=True)

        # Stop tracking this for autoclose
        monitor_closer.unregister(self._monitor_id)
        self.enabled = False

        logger.info('''Finished writing results. You can upload them to the scoreboard via gym.upload(%r)''', self.directory)

    def _before_step(self, action):
        if not self.enabled: return
        self.stats_recorder.before_step(action)

    def _after_step(self, observation, reward, done, info):
        if not self.enabled: return done

        if done and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
            self.reset_video_recorder()
            self.episode_id += 1
            self._flush()

        # Record stats
        self.stats_recorder.after_step(observation, reward, done, info)
        # Record video
        self.video_recorder.capture_frame()

        return done

    def _before_reset(self):
        if not self.enabled: return
        self.stats_recorder.before_reset()

    def _after_reset(self, observation):
        if not self.enabled: return

        # Reset the stat count
        self.stats_recorder.after_reset(observation)

        self.reset_video_recorder()

        # Bump *after* all reset activity has finished
        self.episode_id += 1

        self._flush()
