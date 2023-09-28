import matplotlib.pyplot as plt
import matplotlib.dates as plot_dates
import numpy as np
import pandas as pd

from raw_data_manager.models import EventParameters


def class_is_normal(class_id: int) -> bool:
    """Verifies if a given class is of normal type.

    Parameters
    ----------
    class_id: int
        The class identifier.

    Returns
    -------
    bool
        True if the class is normal, False otherwise.
    """
    return class_id == 0


def display_entire_event(
    event_data: pd.DataFrame,
    class_type_name: str,
    show_interpolate: bool = False,
    interpolate_type: bool = "linear",
    language: str = "pt",
) -> None:
    """Display the entire event data with variations of variables over time.

    Parameters
    ----------
    event_data: pd.DataFrame
        The event data as a DataFrame.
    class_type_name: str
        The name of the class type associated with the event.
    show_interpolate: bool, optional
        Whether to show interpolated values, by default False.
    interpolate_type: bool, optional
        The type of pandas.DataFrame.interpolate method to use, by default "linear".
    language: str, optional
        The language for plot labels, "pt" or "en", by default "pt".
    """

    # Create subplots
    fig, axs = plt.subplots(
        len(EventParameters.event_num_attribs), 1, sharex="col", figsize=(15, 22)
    )
    fig.suptitle(
        f"Variação das variáveis de um evento com classe {class_type_name} ao longo do tempo."
        if language == "pt"
        else f"Variation of the variables of an event of type {class_type_name} along time.",
        fontsize=20,
    )

    # Populate subplots
    for column_name, ax in zip(EventParameters.event_num_attribs, axs.flatten()):
        generate_event_variable_subplot(
            event_data[column_name],
            event_data.index,
            event_data["class"],
            class_type_name,
            show_interpolate=show_interpolate,
            interpolate_type=interpolate_type,
            language=language,
            ax=ax,
        )
    plt.title(
        f"Variação das variáveis de um evento com classe {class_type_name} ao longo do tempo."
        if language == "pt"
        else f"Variation of the variables of an event of type {class_type_name} along time."
    )
    plt.subplots_adjust(top=30)

    # Add legends
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
        text.set_fontsize(14)

    # Show figure
    plt.tight_layout()
    fig.subplots_adjust(
        top=0.90,
    )
    plt.show()


def generate_event_variable_subplot(
    series: pd.Series,
    timestamp: pd.Series,
    class_type: pd.Series,
    class_type_name: str,
    ax: plt.axes,
    show_interpolate: bool = False,
    interpolate_type: str = "linear",
    language: str = "pt",
) -> None:
    """Plots data of a single variable grouped by class type and data availability.

    Parameters
    ----------
    series: pd.Series
        The data series for the variable.
    timestamp: pd.Series
        The timestamps associated with the data.
    class_type: pd.Series
        The class types associated with the data.
    class_type_name: str
        The name of the class type associated with the event.
    ax: plt.axes
        The matplotlib axes to plot on.
    show_interpolate: bool, optional
        Whether to show interpolated values, by default False.
    interpolate_type: bool, optional
        The type of pandas.DataFrame.interpolate method to use, by default "linear".
    language: str, optional
        The language for plot labels, "pt" or "en", by default "pt".
    """
    timestamp = pd.to_datetime(timestamp)
    variable_name = series.name
    units = (
        "[Pa]"
        if variable_name.startswith("P")
        else "[ºC]"
        if variable_name.startswith("T")
        else "[sm³/s]"
        if variable_name.startswith("Q")
        else ""
    )

    # Plot interpolation function on sections with missing data
    if show_interpolate:
        fc_flux_interpolation = series.interpolate(method=interpolate_type)
        fc_flux_interpolation = np.where(series.isnull(), fc_flux_interpolation, np.nan)
        ax.plot(
            timestamp,
            fc_flux_interpolation,
            color="#3498db",
            linestyle="dotted",
            label="Interpolação" if language == "pt" else "Interpolation",
        )

    # Separating series into groups by class_type and data availalibity
    fc_flux_normal = np.where(class_is_normal(class_type), series, np.nan)
    fc_flux_anomaly_transient = np.where(
        class_type.between(101, 108, inclusive="both"), series, np.nan
    )
    fc_flux_anomaly_permanent = np.where(
        class_type.between(1, 8, inclusive="both"), series, np.nan
    )
    fc_flux_unknown_status = np.where(
        class_type.isna() & series.notna(), series, np.nan
    )
    fc_flux_no_data = np.where(series.isna(), series.min(), np.nan)

    # Plotting series data colored by class_type and data availalibity
    ax.plot(
        timestamp,
        fc_flux_normal,
        color="#2ecc71",
        marker=".",
        markersize=3,
        label="Escoamento em estado normal"
        if language == "pt"
        else "Flow in normal state",
    )
    ax.plot(
        timestamp,
        fc_flux_anomaly_transient,
        color="#f1c40f",
        marker=".",
        markersize=3,
        label=f"Escoamento em estado {class_type_name} transiente"
        if language == "pt"
        else f"Flow in transient {class_type_name} state",
    )
    ax.plot(
        timestamp,
        fc_flux_anomaly_permanent,
        color="#e74c3c",
        marker=".",
        markersize=3,
        label=f"Escoamento em estado {class_type_name} permanente"
        if language == "pt"
        else f"Flow in permanent {class_type_name} state",
    )
    ax.plot(
        timestamp,
        fc_flux_unknown_status,
        color="#7f8c8d",
        marker=".",
        markersize=3,
        label="Escoamento em estado desconhecido"
        if language == "pt"
        else "Flow in unknown state",
    )
    ax.plot(
        timestamp,
        fc_flux_no_data,
        color="#d3d3d3",
        marker=".",
        markersize=5,
        label="Escoamento sem dados" if language == "pt" else "Flow missing data",
    )

    # Plot labelling
    ax.set_xlabel("Tempo [H:M [d/m/Y]]" if language == "pt" else "Time [H:M [d/m/Y]]")
    ax.set_ylabel(
        f"Valor de {series.name} {units}"
        if language == "pt"
        else f"Value of {series.name} {units}"
    )
    ax.set_title(
        f"Variação da variável {series.name}."
        if language == "pt"
        else f"Variation of {series.name}."
    )
    ax.xaxis.set_major_formatter(plot_dates.DateFormatter("%H:%M [%d/%m/%Y]"))
