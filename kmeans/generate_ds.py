import random
import click


def generate_dataset(n, d, l, u, out):
    with open(out, 'wt') as fd:
        for i in range(n):
            instances = [str(random.uniform(l, u)) for _ in range(d)]
            instances_string = ' '.join(instances) + '\n'
            fd.write(instances_string)

    click.echo('Generated dataset. Params: n = %d, d = %d, l = %f, u = %f, out = %s' % (n, d, l, u, out))


@click.command()
@click.option('--n', required=True, type=click.INT)
@click.option('--d', required=True, type=click.INT)
@click.option('--l', required=True, type=click.FLOAT)
@click.option('--u', required=True, type=click.FLOAT)
@click.option('--out', required=True)
def main(n, d, l, u, out):
    generate_dataset(n, d, l, u, out)


if __name__ == '__main__':
    main()
