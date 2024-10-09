#!/usr/bin/env python3
import click
import subprocess
import sys

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli():
    """Genesys CLI tool for running configuration, node, and evolution commands."""
    pass

@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),add_help_option=False)
@click.pass_context
def cfg(ctx):
    """Upload or download configurations"""
    cmd = [sys.executable, '-m', 'bin.pages.config'] + ctx.args
    subprocess.run(cmd)

@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),add_help_option=False)
@click.pass_context
def node(ctx):
    """Run the listener node"""
    cmd = [sys.executable, '-m', 'bin.pages.listen'] + ctx.args
    subprocess.run(cmd)

@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),add_help_option=False)
@click.pass_context
def evo(ctx):
    """Run the evolution loop"""
    cmd = [sys.executable, '-m', 'bin.pages.evolve'] + ctx.args
    subprocess.run(cmd)

@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True), add_help_option=False)
@click.pass_context
def gui(ctx):
    """Run the Streamlit GUI"""
    cmd = ['streamlit', 'run', 'bin/app.py'] + ctx.args
    subprocess.run(cmd)

if __name__ == '__main__':
    cli()