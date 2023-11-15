import sys
import getopt

def main(argv):
    batch_size = 4
    learning_rate = 6.7040e-6
    warmup_steps = 70
    opts, args = getopt.getopt(argv, 'hb:l:w:', ['batch_size=', 'learning_rate=', 'warmup_steps='])

    for opt, arg in opts:
        if opt == '-h':
            print('main.py -b <batch size> -l <learning rate> -w <warmup steps>')
            print('main.py --batch_size <batch size> --learning_rate <learning rate> --warmup_steps <warmup steps>')
            sys.exit()
        elif opt in ('-b', '--batch_size'):
            batch_size = arg
        elif opt in ('-l', '--learning_rate'):
            learning_rate = arg
        elif opt in ('-w', '--warmup_steps'):
            warmup_steps = arg

    print(f'Batch size: {batch_size}; Learning rate: {learning_rate}; Warmup steps: {warmup_steps}')

    from setup import setup
    setup(batch_size, learning_rate, warmup_steps)

if __name__ == '__main__':
   main(sys.argv[1:])