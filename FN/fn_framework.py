import os
import io
import re
from collections import namedtuple
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as K
from PIL import Image
import matplotlib.pyplot as plt


Experience = namedtuple("Experience",
                        ["s", "a", "r", "n_s", "d"])  # 経験を格納するnamedtuple


class FNAgent():
    # モデル（価値関数）の保存・ロード、初期化
    # モデル（価値関数）を使った価値予測
    # モデル（価値関数）の学習（パラメータ更新）
    # モデル（価値関数）を使った戦略実施（Epsilon-greedy）
    # play（学習済みのモデルの動作確認）

    def __init__(self, epsilon, actions):
        self.epsilon = epsilon
        self.actions = actions
        self.model = None
        self.estimate_probs = False
        self.initialized = False

    def save(self, model_path):  # model_pathにモデルの保存
        self.model.save(model_path, overwrite=True, include_optimizer=False)

    @classmethod  # クラスメソッドの宣言（インスタンスを作らずにクラスから直接メソッドを呼び出せる）
    def load(cls, env, model_path, epsilon=0.0001):  # 環境とパスを入力として、agentとモデルをロード（playで使う）
        actions = list(range(env.action_space.n))  # 環境のaction_space数からaction番号リスト（actions）を作成
        agent = cls(epsilon, actions)  # actionsからagentのロード
        agent.model = K.models.load_model(model_path)  # model_pathからモデルのロード（agent.model）
        agent.initialized = True
        return agent

    def initialize(self, experiences):  # モデル（価値関数）の初期化（モデル構築、正規化）
        raise NotImplementedError("You have to implement initialize method.")

    def estimate(self, s):  # モデル（価値関数）を使った価値予測（policy内で使用）
        raise NotImplementedError("You have to implement estimate method.")

    def update(self, experiences, gamma):  # モデル（価値関数）の学習（パラメータ更新）
        raise NotImplementedError("You have to implement update method.")

    def policy(self, s):
        # モデル（価値関数）を使った戦略実施（Epsilon-greedy）
        # 訓練（train_loop）と動作確認のためのシミュレーション（play）の両方で使う
        if np.random.random() < self.epsilon or not self.initialized:  # epsilon以下かinitialized=Fの場合、離散一様ランダム
            return np.random.randint(len(self.actions))
        else:
            estimates = self.estimate(s)  # モデル（価値関数）を使った価値予測
            if self.estimate_probs:  # 予測対象が行動確率の場合
                action = np.random.choice(self.actions,  # estimate()で予測した行動確率でランダム
                                          size=1, p=estimates)[0]
                return action
            else:
                return np.argmax(estimates)  # 予測対象が行動確率でない場合、最大確率の行動を選択

    def play(self, env, episode_count=5, render=True):
        # モデル（価値関数）を使ってepisode_count分シミュレーション実施（学習済みのモデルの動作確認のため）
        for e in range(episode_count):
            s = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if render:
                    env.render()
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)
                episode_reward += reward
                s = n_state
            else:
                print("Get reward {}.".format(episode_reward))


class Trainer():  # Agentにデータを与えて訓練（train_loop）

    def __init__(self, buffer_size=1024, batch_size=32,
                 gamma=0.9, report_interval=10, log_dir=""):
        self.buffer_size = buffer_size  # 経験を格納するexperiencesのサイズ
        self.batch_size = batch_size  # １回の学習のためにexperiencesから取り出すデータの大きさ（experience replayのため）
        self.gamma = gamma
        self.report_interval = report_interval  # 未確認
        self.logger = Logger(log_dir, self.trainer_name)  # Loggerの起動？（log_dirにTrainer名でログ保存）
        self.experiences = deque(maxlen=buffer_size)  # 左右からappend可能なコンテナ, 古い順に削除される, maxlenでサイズ指定
        self.training = False
        self.training_count = 0
        self.reward_log = []

    @property
    def trainer_name(self):  # Trainerの名称をプロパティとして設定（その際、英数字のフォーマットを変更）
        class_name = self.__class__.__name__
        snaked = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        snaked = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snaked).lower()
        snaked = snaked.replace("_trainer", "")
        return snaked

    def train_loop(self, env, agent, episode=200, initial_count=-1,
                   render=False, observe_interval=0):
        # episode分だけ学習ループを実施
        # 各episodeでは、buffer_size分experiencesが蓄積するか、initial_count分エピソードを消化したら学習開始
        # 学習はstep内のupdate（Agentで定義）で。
        # observe_interval:プレイ中の画面（状態）の格納頻度

        self.experiences = deque(maxlen=self.buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []
        frames = []  # 未確認

        for i in range(episode):
            s = env.reset()
            done = False
            step_count = 0
            self.episode_begin(i, agent)  # エピソード開始
            while not done:
                if render:
                    env.render()
                if self.training and observe_interval > 0 and\  # i)訓練中 + ii)格納頻度>0 + iii)格納のタイミング
                   (self.training_count == 1 or
                    self.training_count % observe_interval == 0):
                    frames.append(s)  # framesにsを追加

                a = agent.policy(s)
                n_state, reward, done, info = env.step(a)
                e = Experience(s, a, reward, n_state, done)
                self.experiences.append(e)  # experiencesに経験Experienceを追加

                if not self.training and \  # i)訓練外 + ii)experiencesのサイズがbuffer_sizeに到達
                   len(self.experiences) == self.buffer_size:
                    self.begin_train(i, agent)  # 訓練開始
                    self.training = True

                self.step(i, step_count, agent, e)
                    # experiencesからbatch_size分のbatchをサンプリング、update（学習、Agent内で定義）を実行

                s = n_state
                step_count += 1

            else:  # done=Trueの場合
                self.episode_end(i, step_count, agent)  # 各Trainerで定義

                if not self.training and \  # i)訓練外 + ii)エピソード数がinitial_countに到達
                   initial_count > 0 and i >= initial_count:
                    self.begin_train(i, agent)  # 訓練開始
                    self.training = True

                if self.training:  # 訓練中
                    if len(frames) > 0:  # framesの長さが正
                        self.logger.write_image(self.training_count,  # Logger()で定義
                                                frames)
                        frames = []  # frames初期化
                    self.training_count += 1  # training_count+1

    def episode_begin(self, episode, agent):  # エピソード開始
        pass

    def begin_train(self, episode, agent):  # 訓練開始
        pass

    def step(self, episode, step_count, agent, experience):
        # experiencesからbatch_size分のbatchをサンプリング、update（学習）を実行
        pass

    def episode_end(self, episode, step_count, agent):  # エピソード終了
        pass

    def is_event(self, count, interval):  # 未確認
        return True if count != 0 and count % interval == 0 else False

    def get_recent(self, count):  # 直近のcount分の経験を返す
        recent = range(len(self.experiences) - count, len(self.experiences))
        return [self.experiences[i] for i in recent]


class Observer():  # 環境envのラッパー。transformで、envから得られる状態をエージェントが扱いやすい形に変換

    def __init__(self, env):
        self._env = env

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        return self.transform(self._env.reset())

    def render(self):
        self._env.render(mode="human")

    def step(self, action):  # Trainerのstepではなく、_env（=env）のstep
        n_state, reward, done, info = self._env.step(action)
        return self.transform(n_state), reward, done, info

    def transform(self, state):  # envから得られる状態をエージェントが扱いやすい形に変換
        raise NotImplementedError("You have to implement transform method.")


class Logger():

    def __init__(self, log_dir="", dir_name=""):
        self.log_dir = log_dir
        if not log_dir:  # log_dirの指定がなければ、現在のディレクトリ/logsを指定
            self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
        if not os.path.exists(self.log_dir):  # log_dirに相当するディレクトリがなければ作成
            os.mkdir(self.log_dir)

        if dir_name: # dir_nameの指定があれば、log_dir/dir_name
            self.log_dir = os.path.join(self.log_dir, dir_name)
            if not os.path.exists(self.log_dir):  # log_dirに相当するディレクトリがなければ作成
                os.mkdir(self.log_dir)

        self._callback = tf.compat.v1.keras.callbacks.TensorBoard(
                            self.log_dir)

    @property
    def writer(self):
        return self._callback.writer

    def set_model(self, model):
        self._callback.set_model(model)

    def path_of(self, file_name):
        return os.path.join(self.log_dir, file_name)

    def describe(self, name, values, episode=-1, step=-1):
        mean = np.round(np.mean(values), 3)
        std = np.round(np.std(values), 3)
        desc = "{} is {} (+/-{})".format(name, mean, std)
        if episode > 0:
            print("At episode {}, {}".format(episode, desc))
        elif step > 0:
            print("At step {}, {}".format(step, desc))

    def plot(self, name, values, interval=10):
        indices = list(range(0, len(values), interval))
        means = []
        stds = []
        for i in indices:
            _values = values[i:(i + interval)]
            means.append(np.mean(_values))
            stds.append(np.std(_values))
        means = np.array(means)
        stds = np.array(stds)
        plt.figure()
        plt.title("{} History".format(name))
        plt.grid()
        plt.fill_between(indices, means - stds, means + stds,
                         alpha=0.1, color="g")
        plt.plot(indices, means, "o-", color="g",
                 label="{} per {} episode".format(name.lower(), interval))
        plt.legend(loc="best")
        plt.show()

    def write(self, index, name, value):
        summary = tf.compat.v1.Summary()
        summary_value = summary.value.add()
        summary_value.tag = name
        summary_value.simple_value = value
        self.writer.add_summary(summary, index)
        self.writer.flush()

    def write_image(self, index, frames):
        # Deal with a 'frames' as a list of sequential gray scaled image.
        last_frames = [f[:, :, -1] for f in frames]
        if np.min(last_frames[-1]) < 0:
            scale = 127 / np.abs(last_frames[-1]).max()
            offset = 128
        else:
            scale = 255 / np.max(last_frames[-1])
            offset = 0
        channel = 1  # gray scale
        tag = "frames_at_training_{}".format(index)
        values = []

        for f in last_frames:
            height, width = f.shape
            array = np.asarray(f * scale + offset, dtype=np.uint8)
            image = Image.fromarray(array)
            output = io.BytesIO()
            image.save(output, format="PNG")
            image_string = output.getvalue()
            output.close()
            image = tf.compat.v1.Summary.Image(
                        height=height, width=width, colorspace=channel,
                        encoded_image_string=image_string)
            value = tf.compat.v1.Summary.Value(tag=tag, image=image)
            values.append(value)

        summary = tf.compat.v1.Summary(value=values)
        self.writer.add_summary(summary, index)
        self.writer.flush()
