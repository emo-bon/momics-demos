"""Script for visualizing benchmark results using Plotly.

To use this script, run:

```shell
.venv/bin/python -m scripts.plot_bars
```

Benchmarks inspired by (took over from) fireducks TPC-H benchmark under Apache License 2.0
https://fireducks-dev.github.io/docs/benchmarks/#2-tpc-h-benchmark

This file is a copy of the file scripts/plot_bars.py from the fireducks project (2025-02-06)
https://github.com/fireducks-dev/polars-tpch/blob/fireducks/queries/pandas/utils.py

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import plotly.express as px
import polars as pl

from settings import Settings

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

    from settings import IoType

settings = Settings()


COLORS = {
    "polars": "#0075FF",
    "polars-eager": "#00B4D8",
    "duckdb": "#80B9C8",
    "pyspark": "#C29470",
    "dask": "#77D487",
    "pandas": "#2B8C5D",
    "fireducks": "#5F9EA0",
    "modin": "#50B05F",
}

SOLUTION_NAME_MAP = {
    "polars": "Polars",
    "polars-eager": "Polars - eager",
    "duckdb": "DuckDB",
    "pandas": "pandas",
    "fireducks": "FireDucks",
    "dask": "Dask",
    "modin": "Modin",
    "pyspark": "PySpark",
}

Y_LIMIT_MAP = {
    "skip": 15.0,
    "parquet": 20.0,
    "csv": 25.0,
    "feather": 20.0,
}
LIMIT = settings.plot.y_limit or Y_LIMIT_MAP[settings.run.io_type]


def main() -> None:
    pl.Config.set_tbl_rows(100)
    df = prep_data()
    plot(df)


def prep_data() -> pl.DataFrame:
    lf = pl.scan_csv(settings.paths.timings / settings.paths.timings_filename)

    # Scale factor not used at the moment
    lf = lf.drop("scale_factor")

    # Select timings with the right IO type
    lf = lf.filter(pl.col("io_type") == settings.run.io_type).drop("io_type")

    # Select relevant queries
    lf = lf.filter(pl.col("query_number") <= settings.plot.n_queries)

    # Get the last timing entry per solution/version/query combination
    lf = lf.group_by("solution", "version", "query_number").last()

    # Insert missing query entries
    groups = lf.select("solution", "version").unique()
    queries = pl.LazyFrame({"query_number": range(1, settings.plot.n_queries + 1)})
    groups_queries = groups.join(queries, how="cross")
    lf = groups_queries.join(lf, on=["solution", "version", "query_number"], how="left")
    lf = lf.with_columns(pl.col("duration[s]").fill_null(0))

    # Order the groups
    solutions_in_data = lf.select("solution").collect().to_series().unique()
    solution = pl.LazyFrame({"solution": [s for s in COLORS if s in solutions_in_data]})
    lf = solution.join(lf, on=["solution"], how="left")

    # Make query number a string
    lf = lf.with_columns(pl.format("Q{}", "query_number").alias("query")).drop(
        "query_number"
    )

    return lf.select("solution", "version", "query", "duration[s]").collect()


def plot(df: pl.DataFrame) -> Figure:
    """Generate a Plotly Figure of a grouped bar chart displaying benchmark results."""
    x = df.get_column("query")
    y = df.get_column("duration[s]")

    group = df.select(
        pl.format("{} ({})", pl.col("solution").replace(SOLUTION_NAME_MAP), "version")
    ).to_series()

    # build plotly figure object
    color_seq = [c for (s, c) in COLORS.items() if s in df["solution"].unique()]

    fig = px.histogram(
        x=x,
        y=y,
        color=group,
        barmode="group",
        template="plotly_white",
        color_discrete_sequence=color_seq,
    )

    fig.update_layout(
        title={
            "text": get_title(settings.run.io_type),
            "y": 0.95,
            "yanchor": "top",
        },
        bargroupgap=0.1,
        # paper_bgcolor="rgba(41,52,65,1)",
        xaxis_title="Query",
        yaxis_title="Seconds",
        yaxis_range=[0, LIMIT],
        # plot_bgcolor="rgba(41,52,65,1)",
        margin={"t": 150},
        legend={
            "title": "",
            "orientation": "h",
            "xanchor": "center",
            "yanchor": "top",
            "x": 0.5,
        },
    )

    add_annotations(fig, LIMIT, df)

    write_plot_image(fig)

    # display the object using available environment context
    if settings.plot.show:
        fig.show()


def get_title(io_type: IoType) -> str:
    if io_type == "skip":
        title = "Runtime excluding data read from disk"
    else:
        file_type_map = {"parquet": "Parquet", "csv": "CSV", "feather": "Feather"}
        file_type_formatted = file_type_map[io_type]
        title = f"Runtime including data read from disk ({file_type_formatted})"

    subtitle = "(lower is better)"

    return f"{title}<br><i>{subtitle}<i>"


def add_annotations(fig: Any, limit: float, df: pl.DataFrame) -> None:
    # order of solutions in the file
    # e.g. ['polar', 'pandas']
    bar_order = (
        df.get_column("solution")
        .unique(maintain_order=True)
        .to_frame()
        .with_row_index()
    )

    # we look for the solutions that surpassed the limit
    # and create a text label for them
    df = (
        df.filter(pl.col("duration[s]") > limit)
        .with_columns(
            pl.format(
                "{} took {}s", "solution", pl.col("duration[s]").cast(pl.Int32)
            ).alias("labels")
        )
        .join(bar_order, on="solution")
        .group_by("query")
        .agg(pl.col("labels"), pl.col("index").min())
        .with_columns(pl.col("labels").list.join(",\n"))
    )

    # then we create a dictionary similar to something like this:
    #     anno_data = {
    #         "q1": "label",
    #         "q3": "label",
    #     }
    if df.height > 0:
        anno_data = {
            v[0]: v[1]
            for v in df.select("query", "labels")
            .transpose()
            .to_dict(as_series=False)
            .values()
        }
    else:
        # a dummy with no text
        anno_data = {"q1": ""}

    for q_name, anno_text in anno_data.items():
        fig.add_annotation(
            align="right",
            x=q_name,
            y=LIMIT,
            xshift=0,
            yshift=30,
            showarrow=False,
            text=anno_text,
        )


def write_plot_image(fig: Any) -> None:
    path = settings.paths.plots
    if not path.exists():
        path.mkdir()

    file_name = f"plot-io-{settings.run.io_type}.html"
    print(path / file_name)

    fig.write_html(path / file_name)


if __name__ == "__main__":
    main()