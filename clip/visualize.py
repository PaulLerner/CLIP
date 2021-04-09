# coding: utf-8
"""Usage: visualize.py tabulate <results>"""
import json
from tabulate import tabulate
from docopt import docopt


def tabulate_results(results, headers="keys", tablefmt='latex'):
    with open(results, "r") as file:
        results = json.load(file)
    table = []
    for r in results:
        for k, v in r.items():
            v["split"] = k
            table.append(v)
    print(tabulate(table, headers=headers, tablefmt=tablefmt))


def main():
    args = docopt(__doc__)
    if args['tabulate']:
        tabulate_results(args['<results>'])


if __name__ == '__main__':
    main()
