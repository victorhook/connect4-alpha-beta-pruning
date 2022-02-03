# ConnectFour using minmax alpha-beta pruning search algorithm

Setup project:


```
git clone https://github.com/victorhook/connect4-alpha-beta-pruning
cd connect4-alpha-beta-pruning
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

Now the program can be run by:
```
python main.py --local --rounds 10
```

to run it locally, or:

```
python main.py --online --rounds 10
```