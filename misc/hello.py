import time
import argparse

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("counter", type=int, default=0)
    args = argParser.parse_args()
    
    print(f"Hello from python with counter={args.counter} at {time.ctime()}")

