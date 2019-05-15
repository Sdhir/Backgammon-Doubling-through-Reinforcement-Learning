# Backgammon-Doubling-through-Reinforcement-Learning
The objective of this project is to accept a double and continue playing or reject the double and concede the game at the cost of losing point when the opponent offers double.

Train and test data can be downloaded at [Google Drive](https://drive.google.com/drive/folders/16Y1lcwXPP5fuzOuq5hmMd87x0Ro-EL3C?usp=sharing)

Generate train data - Board positions:

Run this file in [GNU Backgammon CLI](https://www.gnu.org/software/gnubg/) (run in multiple CLIs to generate more data)
> load python gen_train_data.py

returns: csv file with board positions {b0,b1-b24,b25}

Generate train data - Board positions:

Run this file in [GNU Backgammon CLI](https://www.gnu.org/software/gnubg/) (run in multiple CLIs to generate more data)
> load python gen_train_data.py

returns: csv file with ground truth and board positions {y_gt,b0,b1-b24,b25}

Training:

Run in terminal
> python train.py

Testing:

Run in terminal
> python test.py

