import click
import random


def filter_colums(input_elems, ignore_columns, delim):
    line_elems = [elem for idx, elem in enumerate(input_elems) if idx not in ignore_columns]
    return delim.join(line_elems)


def subsample_ds(input_file, output_file, n, delim, ignore_columns, mode):
    elems = []

    with open(input_file, 'r') as fd:
        for line in fd:
            elems.append(line.strip().split(delim))

    ds_n = len(elems)
    if not n:
        n = n or ds_n
    else:
        if n >= ds_n:
            raise ValueError('Cannot sample with higher/equal count than N ({} >= {})'.format(n, ds_n))

    if ignore_columns:
        ignore_columns = [int(e) for e in ignore_columns.strip().split(' ')]
    else:
        ignore_columns = []

    with open(output_file, 'w') as fd:
        if mode == 'normal':
            for i in range(n):
                line = filter_colums(elems[i], ignore_columns, delim)
                fd.write(line + '\n')
            return

        if mode == 'random':
            random_elems = random.sample(elems, n)
            for i in range(n):
                line = filter_colums(random_elems[i], ignore_columns, delim)
                fd.write(line + '\n')


mode_choices = click.Choice(['normal', 'random'])


@click.command()
@click.option('-f', '--input-file', required=True)
@click.option('-o', '--output-file', required=True)
@click.option('-d', '--delim', default=' ', required=True)
@click.option('-n', '--no-instances', type=click.INT,
              help='Number of instances to be sampled; must be lower than total instances')
@click.option('-i', '--ignore-columns',
              help="List of columns to be ignored separated with space (e.g '0 3' -> column 0 and 3 will be ignored)")
@click.option('-m', '--mode', type=mode_choices, default='normal',
              help="Mode of sampling: 'normal' will take first 'n' instances; 'random' takes 'n' random instances")
def main(input_file, output_file, no_instances, ignore_columns, mode, delim):
    subsample_ds(input_file, output_file, no_instances, delim, ignore_columns, mode)
    click.echo("Sampled '{}' from '{}' '{}' instances, ignoring columns '{}'; delim '{}' to output '{}'"
               .format(mode, input_file, no_instances, ignore_columns, delim, output_file))


if __name__ == '__main__':
    main()
