import os
import tensorflow as tf
import options import args

def inference(opt):
  print()
  print("[TEST FILES]")
  for file in sorted(os.listdir(opt.test_dir)):
    print("  ", file)
  print()
  print()


if __name__ == "__main__":
  opt = args
  print(opt)

  if opt.mode is 'test' : 
    inference(opt)