import matplotlib.pyplot as plt
import matplotlib.dates as plot_dates
import numpy as np
import pandas as pd

from constants import module_constants


def class_is_normal(class_):
    return class_ == 0


def generate_event_variable_subplot(
    series,
    timestamp,
    class_,
    type_name,
    ax,
    show_interpolate=False,
    interpolate_type="linear",
    lang="pt",
):
    timestamp = pd.to_datetime(timestamp)
    name = series.name
    units = (
        "[Pa]"
        if name.startswith("P")
        else "[ºC]"
        if name.startswith("T")
        else "[sm³/s]"
        if name.startswith("Q")
        else ""
    )

    if show_interpolate:
        fc_flux_interpolation = series.interpolate(method=interpolate_type)
        fc_flux_interpolation = np.where(series.isnull(), fc_flux_interpolation, np.nan)
        ax.plot(
            timestamp,
            fc_flux_interpolation,
            color="#3498db",
            linestyle="dotted",
            label="Interpolação" if lang == "pt" else "Interpolation",
        )

    fc_flux_normal = np.where(class_is_normal(class_), series, np.nan)
    fc_flux_anomaly_transient = np.where(
        class_.between(101, 108, inclusive="both"), series, np.nan
    )
    fc_flux_anomaly_permanent = np.where(
        class_.between(1, 8, inclusive="both"), series, np.nan
    )
    fc_flux_unknown_status = np.where(class_.isna() & series.notna(), series, np.nan)
    fc_flux_no_data = np.where(series.isna(), series.min(), np.nan)

    ax.plot(
        timestamp,
        fc_flux_normal,
        color="#2ecc71",
        marker=".",
        markersize=3,
        label="Escoamento em estado normal" if lang == "pt" else "Flow in normal state",
    )
    ax.plot(
        timestamp,
        fc_flux_anomaly_transient,
        color="#f1c40f",
        marker=".",
        markersize=3,
        label=f"Escoamento em estado {type_name} transiente"
        if lang == "pt"
        else f"Flow in transient {type_name} state",
    )
    ax.plot(
        timestamp,
        fc_flux_anomaly_permanent,
        color="#e74c3c",
        marker=".",
        markersize=3,
        label=f"Escoamento em estado {type_name} permanente"
        if lang == "pt"
        else f"Flow in permanent {type_name} state",
    )
    ax.plot(
        timestamp,
        fc_flux_unknown_status,
        color="#7f8c8d",
        marker=".",
        markersize=3,
        label="Escoamento em estado desconhecido"
        if lang == "pt"
        else "Flow in unknown state",
    )
    ax.plot(
        timestamp,
        fc_flux_no_data,
        color="#d3d3d3",
        marker=".",
        markersize=5,
        label="Escoamento sem dados" if lang == "pt" else "Flow missing data",
    )

    ax.set_xlabel("Tempo [H:M [d/m/Y]]" if lang == "pt" else "Time [H:M [d/m/Y]]")
    ax.set_ylabel(
        f"Valor de {series.name} {units}"
        if lang == "pt"
        else f"Value of {series.name} {units}"
    )
    ax.set_title(
        f"Variação da variável {series.name}."
        if lang == "pt"
        else f"Variation of {series.name}."
    )
    ax.xaxis.set_major_formatter(plot_dates.DateFormatter("%H:%M [%d/%m/%Y]"))


def display_entire_event(
    event, type_name, show_interpolate=False, interpolate_type="linear", lang="pt"
):
    fig, axs = plt.subplots(
        len(module_constants.event_num_attribs), 1, sharex="col", figsize=(15, 22)
    )
    # fig.suptitle(f'Variação das variáveis de um evento com classe {type_name} ao longo do tempo.' if lang == 'pt' else f'Variation of the variables of an event of type {type_name} along time.', fontsize=20)

    for i, (col, ax) in enumerate(
        zip(module_constants.event_num_attribs, axs.flatten())
    ):
        generate_event_variable_subplot(
            event[col],
            event.index,
            event["class"],
            type_name,
            show_interpolate=show_interpolate,
            interpolate_type=interpolate_type,
            lang=lang,
            ax=ax,
        )

    # plt.title(f'Variação das variáveis de um evento com classe {type_name} ao longo do tempo.' if lang == 'pt' else f'Variation of the variables of an event of type {type_name} along time.')
    # plt.subplots_adjust(top=30)
    handles, labels = axs[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=3,
        fancybox=True,
        shadow=True,
    )
    for text in legend.get_texts():
        text.set_fontsize(14)  # Set the desired font size for the legend text

    plt.tight_layout()
    fig.subplots_adjust(
        top=0.90,
    )
    plt.show()
