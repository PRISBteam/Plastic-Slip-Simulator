import argparse
# import DCC_structure

# https://docs.python.org/3/library/argparse.html

def main():
    """ 
    """
    parser = argparse.ArgumentParser(description='Some description')

    parser.add_argument(
        '--DIM',
        action='store',
        default=3,
        type=int,
        choices=[2, 3],
        required=True,
        help='The spatial dimension of the complex (2 or 3)'
    )

    parser.add_argument(
        '--SIZE',
        action='store',
        nargs=3,
        default=[5,5,5],
        type=int,
        required=True,
        help='The number of unit cells'
    )   

    parser.add_argument(
        '--STRUC',
        action='store',
        default='bcc',
        choices=['simple cubic', 'bcc', 'fcc', 'hcp'],
        required=True,
        help='Lattice structure from: simple cubic, bcc, fcc or hcp'
    )  

    # Do some cool stuff
    args = parser.parse_args()
    print(
        'DIM =', args.DIM,
        '\nSIZE =', *args.SIZE,
        '\nSTRUC =', args.STRUC
    )



if __name__ == "__main__":
    main()
