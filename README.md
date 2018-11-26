# bitcoin-rnn
A impementation of training a LSTM network to associate public bitcoin addresses, with private keys.

[  Note, inversion and non-inversion are highly suggested also wif and non-wif data, see mk-prvadd-pair.py, for examples of generating training data. This public example is only for understanding concepts, true production requires huge training sets, and various alternate representations of bitcoin address abstraction.

Reference for theory of this concept to the following paper.

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.

gru-addr2priv.py     - gru example of process ( gru is best )
priv2pub.py          - generic rnn test case
README.md
mk-privaddr-pair.py  - generate training data ( here privaddr-pair.txt ), note in actuality make this HUGE
privaddr-pair.txt    - example training data two column both WIF, first address, second private-key, in this case of 'training' to keep stuff simple WIF is used for both cases, in actuality raw 0-9 decimal should be considered, but in this 'toy' example I use WIF ( compressed hex ) to keep stuff simple

***

Uses for data generated feeds for prime-decomposition, or deep gpu processing of ecdsa; Sage and many tools have additions for large-scale prime data such as SECPK256 ( NSA algo for bitcoin )

