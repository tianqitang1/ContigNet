#!/usr/bin/env python

import os
import pandas as pd
import warnings
import numpy as np
import torch
from . import util, VirusCNN_siamese
import pkgutil
from io import BytesIO
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

warnings.filterwarnings(
    "ignore", message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling."
)


@click.command()
@click.option(
    '--host-dir', '-ho',
    default='demo/host_fasta',
    help='Directory containing host contig sequences in fasta format',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
)
@click.option(
    '--virus-dir', '-vi',
    default='demo/virus_fasta',
    help='Directory containing virus contig sequences in fasta format',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
)
@click.option(
    '--output', '-o',
    default='result.csv',
    help='Path to output file',
    type=click.Path(dir_okay=False, writable=True),
)
@click.option(
    '--cpu',
    is_flag=True,
    help='Force using CPU if specified',
)
@click.option(
    '--show-preview',
    is_flag=True,
    help='Show a preview table of top predictions',
)
def main(host_dir, virus_dir, output, cpu, show_preview):
    """ContigNet: A deep learning based phage-host interaction prediction tool

    Predicts phage-host contig interactions using a convolutional neural network.
    """
    console.print(Panel.fit(
        "[bold cyan]ContigNet[/bold cyan]\n"
        "Phage-host interaction prediction with deep learning",
        border_style="cyan"
    ))

    # Device selection
    if torch.cuda.is_available() and not cpu:
        device = torch.device("cuda")
        console.print(f"[green]✓[/green] Using GPU: [bold]{torch.cuda.get_device_name(0)}[/bold]")
    else:
        if cpu:
            console.print("[yellow]⚠[/yellow] CPU mode explicitly selected")
        else:
            console.print("[yellow]⚠[/yellow] CUDA not available, using CPU")
        device = torch.device("cpu")

    # Load model
    with console.status("[bold green]Loading model...", spinner="dots"):
        model = VirusCNN_siamese.VirusCNN(share_weight=True).to(device)
        model.load_state_dict(torch.load(BytesIO(pkgutil.get_data("ContigNet", "models/model.dict")), map_location=device))
    console.print("[green]✓[/green] Model loaded successfully")

    # Load file lists
    host_list = os.listdir(host_dir)
    host_list.sort()
    host_name_list = [os.path.splitext(i)[0] for i in host_list]
    host_path_list = [os.path.join(host_dir, i) for i in host_list]

    virus_list = os.listdir(virus_dir)
    virus_list.sort()
    virus_name_list = [os.path.splitext(i)[0] for i in virus_list]
    virus_path_list = [os.path.join(virus_dir, i) for i in virus_list]

    console.print(f"[cyan]→[/cyan] Found {len(host_list)} host contig(s)")
    console.print(f"[cyan]→[/cyan] Found {len(virus_list)} virus contig(s)")
    console.print(f"[cyan]→[/cyan] Processing {len(host_list) * len(virus_list)} pair(s)")

    result_df = pd.DataFrame(
        np.zeros((len(host_list), len(virus_list))),
        columns=virus_name_list,
        index=host_name_list,
    )

    # Preserve the CLI output path before we start computing prediction scores
    output_path = output

    # Process predictions
    with torch.no_grad():
        model.eval()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:

            host_task = progress.add_task("[cyan]Processing hosts...", total=len(host_list))

            for i, host_fn in enumerate(host_list):
                host_name = host_name_list[i]
                host_path = host_path_list[i]
                host_onehot = util.fasta2onehot(host_path)

                virus_task = progress.add_task(f"[magenta]  → {host_name[:20]}...", total=len(virus_list))

                for j, virus_fn in enumerate(virus_list):
                    virus_name = virus_name_list[j]
                    virus_path = virus_path_list[j]
                    virus_onehot = util.fasta2onehot(virus_path)

                    try:
                        host_tensor = torch.Tensor(host_onehot).to(device)[None, None, :, :]
                        virus_tensor = torch.Tensor(virus_onehot).to(device)[None, None, :, :]
                        if str(device) != "cpu":
                            score = (
                                torch.sigmoid(model(host_tensor, virus_tensor))
                                .cpu()
                                .numpy()
                                .flatten()[0]
                            )
                        else:
                            score = (
                                torch.sigmoid(model(host_tensor, virus_tensor))
                                .numpy()
                                .flatten()[0]
                            )
                    except RuntimeError as e:  # Fallback in case of out of GPU memory
                        if "CUDA error: out of memory" in str(e):
                            console.print("[yellow]⚠[/yellow] GPU out of memory, falling back to CPU")
                            torch.cuda.empty_cache()
                            model = model.to("cpu")
                            host_tensor = torch.Tensor(host_onehot)[None, None, :, :]
                            virus_tensor = torch.Tensor(virus_onehot)[None, None, :, :]
                            score = (
                                torch.sigmoid(model(host_tensor, virus_tensor))
                                .numpy()
                                .flatten()[0]
                            )
                        else:
                            raise e
                    # Convert numpy scalar to Python float for pandas compatibility
                    result_df.loc[host_name, virus_name] = float(score)
                    progress.update(virus_task, advance=1)

                progress.remove_task(virus_task)
                progress.update(host_task, advance=1)

    # Save results
    result_df.to_csv(output_path)
    console.print(f"[green]✓[/green] Results saved to: [bold]{output_path}[/bold]")

    # Show preview if requested
    if show_preview:
        console.print()

        # Find top predictions
        top_pairs = []
        for host in result_df.index:
            for virus in result_df.columns:
                score = result_df.loc[host, virus]
                top_pairs.append((host, virus, score))

        top_pairs.sort(key=lambda x: x[2], reverse=True)
        top_10 = top_pairs[:min(10, len(top_pairs))]

        # Create preview table
        table = Table(title="Top 10 Predictions", box=box.ROUNDED)
        table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
        table.add_column("Host", style="green")
        table.add_column("Virus", style="magenta")
        table.add_column("Score", justify="right", style="yellow")

        for idx, (host, virus, score) in enumerate(top_10, 1):
            table.add_row(
                str(idx),
                host[:30] + "..." if len(host) > 30 else host,
                virus[:30] + "..." if len(virus) > 30 else virus,
                f"{score:.4f}"
            )

        console.print(table)

    console.print()
    console.print(Panel.fit("[bold green]✓ Analysis complete![/bold green]", border_style="green"))


if __name__ == "__main__":
    main()
