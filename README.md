
# Rock-Paper-Scissors

Rock, paper Scissors is a game made with python3, tensorflow and openCV.
It uses your webcam to recognize your move, and you play against the computer. It features, training, and scores

## Run Locally

First of all, I highly recommend you to use miniforge or anaconda
for projects using tensorflow, since you don't have to install all dependencies globally.
If this is your first tensorflow project, and you don't have tensorflow set up I will refer you to two youtube videos to get you started

**MacOS (M1):**
https://www.youtube.com/watch?v=_CO-ND1FTOU&ab_channel=JeffHeaton

**MacOS (Intel):** https://www.youtube.com/watch?v=LnzgQr14p7s&ab_channel=JeffHeaton

**Windows:** https://www.youtube.com/watch?v=OEFKlRSd8Ic&t=1192s&ab_channel=JeffHeaton

Clone the project

```bash
  git clone https://github.com/Tovborg/tensorflow-rock-paper-scissors.git
```

Go to the project directory

```bash
  cd tensorflow-rock-paper-scissors
```

Install dependencies

```bash
  pip Install -r requirements.txt
```

Start by collecting samples. (more samples = better accuracy)

```bash
  python train.py
```
Now you can start playing

```bash
  python play.py
```
