# Metaverse

This project aims to learn to learn to play video games. That was not a typo: the aim here is meta-learning, or learning to learn. If this works, it would be a major step in the direction of [AGI](https://en.wikipedia.org/wiki/Artificial_general_intelligence).

# Motivation

Most problems in real life can be translated into video games. If we had AI that could use intuition to solve new games quickly, that could go a *long* way towards strong AI.

If things go well, this will be a huge stage of my [meta-learning quest](https://blog.aqnichol.com/2017/04/15/the-meta-learning-quest-part-1/).

# Hypotheses

A feed-forward network without [sgdstore](https://github.com/unixpickle/sgdstore) will probably go a long way. It may be possible to train one network on at least half a dozen games simultaneously. Thus, it is important to gather baselines and ensure that sgdstore and other meta-learning approaches actually help.

I fear that there will be a great amount of overfitting. There are only ~100 games in Universe with reward signals. If I want more data, I may have to find a way to use the games without reward signals.

I also suspect that TRPO will fail for some of the games in Universe. It is possible that transfer learning will actually *help* learn those games, but I am not super optimistic.

To remedy the above problems, I will probably end up taking one of two approaches. One, I may collect demonstration data and use it to perform supervised meta-learning. Two, I may use a per-game critic to leverage actor-critic methods like GAE. However, I do not see myself using anything like A3C; it is unclear to me how well A3C works for long time dependencies.
