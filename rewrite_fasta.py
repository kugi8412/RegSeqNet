import argparse
import os
import numpy as np


def rewrite_fasta(file, outdir=None, name_pos=None):
    # check if there is more than one sequence in the given file
    with open(file, 'r') as f:
        i = 0
        line = f.readline()
        while i < 2 and line:
            if line.startswith('>'):
                i += 1
            line = f.readline()
        if i == 1:
            print('No rewriting was done: given file contains only one sequence.')
            return 1, None
    if outdir is None:
        outdir, name = os.path.split(file)
        namespace, _ = os.path.splitext(name)
        outdir = os.path.join(outdir, namespace)
    if os.path.isdir(outdir):
        num_files = len([el for el in os.listdir(outdir) if el.endswith('.fasta')])
    else:
        os.mkdir(outdir)
        num_files = 0
    if num_files == 0:
        i = 0
        with open(file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    if name_pos is not None:
                        name_pos = [int(el) for el in name_pos]
                        filename = '-'.join([str(la) for la in np.array(line.strip('> \n').split(' '))[name_pos]]) + '.fasta'
                    else:
                        filename = '-'.join(line.strip('> \n').split(' ')[:2]).strip('chr ') + '.fasta'
                    w = open(os.path.join(outdir, filename), 'w')
                    w.write(line)
                    i += 1
                else:
                    w.write(line)
                    w.close()
        print('Based on {} {} sequences were written into separated files in {}'.format(file, i, outdir))
        return i, outdir
    else:
        print('Directory {} with {} fasta files already exists - no rewritting was done.'.format(outdir, num_files))
        return num_files, outdir


def get_names(file, outfile):
    if outfile is None:
        outdir, filename = os.path.split(file)
        filename, ext = os.path.splitext(filename)
        outfile = os.path.join(outdir, filename + '_names.txt')
    w = open(outfile, 'w')
    i = 0
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                line = line.strip().split(' ')
                name = '{}-{}'.format(line[1].lstrip('chr'), line[2])
                w.write(name + '\n')
                i += 1
    w.close()
    print('Names of the {} sequences from {} were saved to {}'.format(i, file, outfile))
    return i, outfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rewrite sequences to separated files')
    parser.add_argument('file', action='store', metavar='FILE', type=str,
                        help='Fasta file or directory with files to rewrite')
    parser.add_argument('method', action='store', type=str, choices=['rewrite', 'get_names'],
                        help='Choose what you want to do with the given file: rewrite sequences to separated files '
                             '(method "rewrite") or write names of sequences to separate file (method "get_names").')
    parser.add_argument('-e', '--extension', action='store', metavar='EXT', type=str, default='fa',
                        help='Extension of the files in the given [PATH] which should be rewritten (method "rewrite"), '
                             'default "fa"')
    parser.add_argument('-o', '--output', action='store', metavar='DIR', type=str, default=None,
                        help='Output directory (for method "rewrite") or file (method "get_names"), '
                             'by default directory of the input file is used (new folder is created) or '
                             'names are saved to file [FILE]_names.txt.')
    args = parser.parse_args()

    if args.method == 'rewrite':
        if os.path.isdir(args.file):
            for f in [el for el in os.listdir(args.file) if os.path.isfile(os.path.join(args.file, el)) and
                      el.endswith(args.extension)]:
                rewrite_fasta(os.path.join(args.file, f), args.output)
        else:
            rewrite_fasta(args.file, args.output)
    elif args.method == 'get_names':
        get_names(args.file, args.output)
