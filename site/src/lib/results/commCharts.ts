const config = {
  background: null,
  font: "Inter, ui-sans-serif, system-ui, sans-serif",
  view: { stroke: null },
  title: {
    fontSize: 17,
    subtitleFontSize: 13,
    subtitleColor: "#50617d",
  },
  axis: {
    domainColor: "#8290a8",
    gridColor: "#e5e8e2",
    labelColor: "#50617d",
    titleColor: "#20304e",
    titleFontWeight: 600,
    labelFontSize: 13,
    titleFontSize: 14,
  },
  legend: {
    labelColor: "#50617d",
    titleColor: "#20304e",
    orient: "bottom",
    direction: "horizontal",
    labelFontSize: 13,
    titleFontSize: 14,
    symbolSize: 110,
  },
  header: {
    labelFontSize: 15,
    titleFontSize: 15,
  },
};

const gpuAxis = {
  field: "gpuCount",
  type: "quantitative",
  title: "GPUs",
  scale: { type: "log", base: 2 },
  axis: { values: [1, 2, 4, 8, 16, 32, 64], format: "d" },
};

// Same compute/transfer/wait colors as the PNG comm figures
// (toolsbench.visualization.comm._COMPUTE_COLOR / _TRANSFER_COLOR / _WAIT_COLOR),
// so the interactive charts and the static figures read as one system.
const computeColor = "#2563eb";
const transferColor = "#f97316";
const waitColor = "#dc2626";

// Each study has exactly 2 sections; every chart below is built once per
// study (not merged) so a panel never has to carry more than 2 sections x
// 2 dimensionalities at once.
type Study = {
  label: string;
  sections: [string, string];
  colors: [string, string];
};

const INFERENCE: Study = {
  label: "Inference",
  sections: ["denoiser", "gradient"],
  colors: ["#168d80", "#f2b46d"],
};

const TRAINING: Study = {
  label: "Training",
  sections: ["forward", "backward"],
  colors: ["#50617d", "#a3423c"],
};

const dimDash = {
  field: "dim",
  type: "nominal",
  title: "Dimensionality",
  scale: {
    domain: ["2D", "3D"],
    range: [
      [1, 0],
      [6, 4],
    ],
  },
};

// Same two-shade-per-section palette as the PNG's _SECTION_SHADES: (dark,
// light) = (compute, comm) for that section, so the two phases stay legible
// on one stacked bar without alpha blending.
const SECTION_SHADES: [string, string][] = [
  ["#1d4ed8", "#93c5fd"], // section 0 (denoiser / forward): dark, light blue
  ["#c2410c", "#fdba74"], // section 1 (gradient / backward): dark, light orange
];

function shareSpec(study: Study) {
  const [s0, s1] = study.sections;
  const segmentDomain = [`${s0} compute`, `${s0} comm`, `${s1} compute`, `${s1} comm`];
  const segmentRange = [
    SECTION_SHADES[0][0],
    SECTION_SHADES[0][1],
    SECTION_SHADES[1][0],
    SECTION_SHADES[1][1],
  ];
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v6.json",
    description: `${study.label}: compute vs communication, scaled to baseline. One stacked bar per GPU count, combining both sections -- same construction as comm_share.png.`,
    title: {
      text: `${study.label}: compute vs communication cost`,
      subtitle:
        "GPU-seconds relative to the smallest GPU count (dashed line = 1.0, ideal linear scaling). One bar per GPU count, both sections stacked together.",
    },
    config,
    facet: {
      column: {
        field: "dim",
        type: "nominal",
        title: "Problem",
      },
    },
    spec: {
      width: 360,
      height: 440,
      layer: [
        {
          mark: { type: "rule", strokeDash: [6, 5], color: "#8290a8" },
          encoding: { y: { datum: 1 } },
        },
        {
          transform: [
            { fold: ["scaledComputeSec", "scaledCommSec"], as: ["phase", "seconds"] },
            {
              calculate: `datum.section + (datum.phase === 'scaledComputeSec' ? ' compute' : ' comm')`,
              as: "segment",
            },
            {
              calculate: `indexof(${JSON.stringify(study.sections)}, datum.section) * 2 + (datum.phase === 'scaledComputeSec' ? 0 : 1)`,
              as: "segmentOrder",
            },
            {
              joinaggregate: [{ op: "sum", field: "seconds", as: "barTotal" }],
              groupby: ["gpuCount"],
            },
            { calculate: "100 * datum.seconds / datum.barTotal", as: "pct" },
          ],
          // band < 1 leaves a visible gap between adjacent GPU-count bars.
          mark: { type: "bar", width: { band: 0.62 } },
          encoding: {
            x: { field: "gpuCount", type: "ordinal", title: "GPUs", sort: "ascending" },
            y: {
              field: "seconds",
              aggregate: "sum",
              stack: "zero",
              title: "GPU-seconds relative to baseline",
            },
            order: { field: "segmentOrder", type: "quantitative" },
            color: {
              field: "segment",
              type: "nominal",
              title: null,
              scale: { domain: segmentDomain, range: segmentRange },
            },
            tooltip: [
              { field: "section", type: "nominal", title: "Section" },
              { field: "gpuCount", type: "quantitative", title: "GPUs" },
              { field: "phase", type: "nominal", title: "Phase" },
              { field: "pct", type: "quantitative", title: "Share of bar", format: ".0f" },
              { field: "seconds", type: "quantitative", title: "Scaled (s)", format: ".3f" },
            ],
          },
        },
        {
          // Percentage-of-bar labels inside each segment -- same as
          // comm_share.png's per-segment annotate() calls (>=3% only).
          transform: [
            { fold: ["scaledComputeSec", "scaledCommSec"], as: ["phase", "seconds"] },
            {
              calculate: `indexof(${JSON.stringify(study.sections)}, datum.section) * 2 + (datum.phase === 'scaledComputeSec' ? 0 : 1)`,
              as: "segmentOrder",
            },
            {
              stack: "seconds",
              groupby: ["gpuCount"],
              sort: [{ field: "segmentOrder" }],
              as: ["segStart", "segEnd"],
            },
            {
              joinaggregate: [{ op: "sum", field: "seconds", as: "barTotal" }],
              groupby: ["gpuCount"],
            },
            { calculate: "100 * datum.seconds / datum.barTotal", as: "pct" },
            { calculate: "(datum.segStart + datum.segEnd) / 2", as: "segMid" },
            { calculate: "format(datum.pct, '.0f') + '%'", as: "pctLabel" },
            { filter: "datum.pct >= 3" },
          ],
          mark: { type: "text", fontSize: 12, fontWeight: "bold", color: "white" },
          encoding: {
            x: { field: "gpuCount", type: "ordinal", sort: "ascending" },
            y: { field: "segMid", type: "quantitative" },
            text: { field: "pctLabel", type: "nominal" },
          },
        },
        {
          // Bar-total label ("Nx" of baseline), same as comm_share.png's
          // bar_total annotation above each bar.
          transform: [
            { fold: ["scaledComputeSec", "scaledCommSec"], as: ["phase", "seconds"] },
            {
              calculate: `indexof(${JSON.stringify(study.sections)}, datum.section) * 2 + (datum.phase === 'scaledComputeSec' ? 0 : 1)`,
              as: "segmentOrder",
            },
            {
              stack: "seconds",
              groupby: ["gpuCount"],
              sort: [{ field: "segmentOrder" }],
              as: ["segStart", "segEnd"],
            },
            { filter: "datum.segmentOrder == 3" },
            { calculate: "format(datum.segEnd, '.2f') + 'x'", as: "totalLabel" },
          ],
          mark: { type: "text", dy: -10, fontSize: 12, fontWeight: "bold", color: "#111827" },
          encoding: {
            x: { field: "gpuCount", type: "ordinal", sort: "ascending" },
            y: { field: "segEnd", type: "quantitative" },
            text: { field: "totalLabel", type: "nominal" },
          },
        },
      ],
    },
  };
}

function absoluteSpec(study: Study) {
  // Same layout as comm.py's _plot_time_breakdown: one panel per section
  // (facet column), 2D and 3D overlaid on the same axes within each panel
  // (solid vs dashed), not split into separate facet columns.
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v6.json",
    description: `${study.label}: absolute compute/transfer/wait time vs GPU count, log scale, 2D and 3D overlaid -- same layout as time_breakdown.png.`,
    title: {
      text: `${study.label}: what's driving that growth?`,
      subtitle:
        "Critical-path compute, transfer, and wait time per section, log scale. Solid = 2D, dashed = 3D.",
    },
    config,
    facet: {
      column: {
        field: "section",
        type: "nominal",
        title: null,
        sort: study.sections,
      },
    },
    spec: {
      width: 360,
      height: 400,
      transform: [
        { fold: ["computeMaxSec", "transferSec", "waitSec"], as: ["phase", "seconds"] },
        { filter: "datum.seconds > 0" },
      ],
      mark: { type: "line", point: { size: 40 }, strokeWidth: 2.2 },
      encoding: {
        x: gpuAxis,
        y: {
          field: "seconds",
          type: "quantitative",
          title: "Time (s, log)",
          scale: { type: "log" },
        },
        color: {
          field: "phase",
          type: "nominal",
          title: null,
          scale: {
            domain: ["computeMaxSec", "transferSec", "waitSec"],
            range: [computeColor, transferColor, waitColor],
          },
          legend: {
            labelExpr:
              "datum.label === 'computeMaxSec' ? 'compute (critical path)' : datum.label === 'transferSec' ? 'transfer' : 'wait'",
          },
        },
        strokeDash: dimDash,
        tooltip: [
          { field: "section", type: "nominal", title: "Section" },
          { field: "dim", type: "nominal", title: "Dim" },
          { field: "gpuCount", type: "quantitative", title: "GPUs" },
          { field: "phase", type: "nominal", title: "Phase" },
          { field: "seconds", type: "quantitative", title: "Seconds", format: ".4f" },
        ],
      },
    },
  };
}

// All four sections in one chart: reuse the same per-section colors already
// established for the (now-merged) per-study wait charts.
const ALL_SECTIONS = [...INFERENCE.sections, ...TRAINING.sections];
const ALL_SECTION_COLORS = [...INFERENCE.colors, ...TRAINING.colors];

function combinedWaitSpec() {
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v6.json",
    description:
      "All four sections (denoiser, gradient, forward, backward): wait time as a percentage of compute time, vs GPU count, symlog scale.",
    title: {
      text: "Are the GPUs actually staying balanced?",
      subtitle:
        "Wait time as a share of useful compute, all sections overlaid. Symlog scale -- values span 0% to 300%+.",
    },
    config,
    facet: {
      column: {
        field: "dim",
        type: "nominal",
        title: "Problem",
      },
    },
    spec: {
      width: 380,
      height: 460,
      layer: [
        {
          mark: { type: "rule", strokeDash: [6, 5], color: "#8290a8" },
          encoding: { y: { datum: 0 } },
        },
        {
          mark: { type: "line", point: { size: 40 }, strokeWidth: 2.4 },
          encoding: {
            x: gpuAxis,
            y: {
              field: "waitPct",
              type: "quantitative",
              title: "Wait / compute (%)",
              scale: { type: "symlog" },
            },
            color: {
              field: "section",
              type: "nominal",
              title: "Section",
              scale: { domain: ALL_SECTIONS, range: ALL_SECTION_COLORS },
            },
            tooltip: [
              { field: "study", type: "nominal", title: "Study" },
              { field: "section", type: "nominal", title: "Section" },
              { field: "gpuCount", type: "quantitative", title: "GPUs" },
              {
                field: "waitPct",
                type: "quantitative",
                title: "Wait / compute (%)",
                format: ".1f",
              },
              { field: "lbPct", type: "quantitative", title: "Load balance (%)", format: ".1f" },
            ],
          },
        },
        {
          // On-chart value labels, since a symlog scale is hard to read exactly by eye.
          transform: [{ calculate: "format(datum.waitPct, '.0f')", as: "waitLabel" }],
          mark: { type: "text", dy: -10, fontSize: 11, fontWeight: "bold", color: "#111827" },
          encoding: {
            x: gpuAxis,
            y: { field: "waitPct", type: "quantitative", scale: { type: "symlog" } },
            text: { field: "waitLabel", type: "nominal" },
          },
        },
      ],
    },
  };
}

// lbCombinedPct/commEPct are duplicated across a study's sections (they're
// already summed across sections upstream) -- keep one section per study so
// each metric draws as a single line, not one overlapping line per section.
const _ONE_SECTION_PER_STUDY_FILTER =
  "(datum.study === 'inference' && datum.section === 'denoiser') || (datum.study === 'training' && datum.section === 'forward')";

// Both metric charts below draw the same shape: one solid line per inference
// section, plus one dashed line per study for the already-combined value.
// Only inference's sections are drawn individually -- training's forward and
// backward track each other closely enough that two extra near-identical
// lines add clutter without adding a story, so training appears as its
// combined line alone.
// The combined lines are aggregates, not two more sections, so they get greys
// rather than another hue -- and the study colors used elsewhere (#168d80 /
// #a3423c) are already taken by denoiser and backward.
const _COMBINED_SERIES = ["inference (combined)", "training (combined)"];
const _COMBINED_COLORS = ["#111827", "#8b8f98"];

const _seriesColor = {
  field: "series",
  type: "nominal",
  title: "Series",
  scale: {
    domain: [...INFERENCE.sections, ..._COMBINED_SERIES],
    range: [...INFERENCE.colors, ..._COMBINED_COLORS],
  },
  // Default labelLimit clips "inference (combined)" to "inference (com...".
  legend: { labelLimit: 220 },
};

// Labels on the combined lines only -- adding them to every section line too
// would be unreadable; section values live in the tooltip and the data table.
// One layer per study, offset in opposite directions: the two combined lines
// sit on top of each other at low GPU counts, so a shared offset stacks their
// labels illegibly.
const _combinedLabelLayers = (combinedField: string) =>
  ([
    ["inference", "denoiser", -11, "#111827"],
    ["training", "forward", 17, "#6b7280"],
  ] as const).map(([study, section, dy, color]) => ({
    transform: [
      { filter: `datum.study === '${study}' && datum.section === '${section}'` },
      { calculate: `format(datum.${combinedField}, '.0f')`, as: "pctLabel" },
    ],
    mark: { type: "text", dy, fontSize: 11, fontWeight: "bold", color },
    encoding: {
      x: gpuAxis,
      y: { field: combinedField, type: "quantitative" },
      text: { field: "pctLabel", type: "nominal" },
    },
  }));

function _sectionPlusCombinedSpec({
  sectionField,
  combinedField,
  sectionTitle,
  combinedTitle,
  title,
  subtitle,
}: {
  sectionField: string;
  combinedField: string;
  sectionTitle: string;
  combinedTitle: string;
  title: string;
  subtitle: string;
}) {
  // Values sit in a high band except where one section collapses -- forcing
  // the axis to include 0 buries the rest of the variation in a sliver.
  const yScale = { zero: false, nice: true };
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v6.json",
    description: `${title} vs GPU count, per section and combined, inference and training overlaid.`,
    title: { text: title, subtitle },
    config,
    facet: {
      column: {
        field: "dim",
        type: "nominal",
        title: "Problem",
      },
    },
    spec: {
      width: 380,
      height: 460,
      layer: [
        {
          // One line per inference section, from that section's own value.
          transform: [
            { filter: "datum.study === 'inference'" },
            { calculate: "datum.section", as: "series" },
          ],
          mark: { type: "line", point: { size: 40 }, strokeWidth: 1.8 },
          encoding: {
            x: gpuAxis,
            y: { field: sectionField, type: "quantitative", title: "%", scale: yScale },
            color: _seriesColor,
            tooltip: [
              { field: "study", type: "nominal", title: "Study" },
              { field: "section", type: "nominal", title: "Section" },
              { field: "gpuCount", type: "quantitative", title: "GPUs" },
              { field: sectionField, type: "quantitative", title: sectionTitle, format: ".1f" },
              { field: combinedField, type: "quantitative", title: combinedTitle, format: ".1f" },
            ],
          },
        },
        {
          // One line per study, from the already-combined value (duplicated
          // across that study's sections upstream, hence the filter).
          transform: [
            { filter: _ONE_SECTION_PER_STUDY_FILTER },
            { calculate: "datum.study + ' (combined)'", as: "series" },
          ],
          mark: { type: "line", point: { size: 40 }, strokeWidth: 3, strokeDash: [6, 3] },
          encoding: {
            x: gpuAxis,
            y: { field: combinedField, type: "quantitative", scale: yScale },
            // The dash pattern lives on the mark, not in an encoding: a
            // strokeDash channel makes Vega union its legend with the color
            // one, and suppressing that union takes the color legend with it.
            color: _seriesColor,
            tooltip: [
              { field: "study", type: "nominal", title: "Study" },
              { field: "gpuCount", type: "quantitative", title: "GPUs" },
              { field: combinedField, type: "quantitative", title: combinedTitle, format: ".1f" },
            ],
          },
        },
        ..._combinedLabelLayers(combinedField),
      ],
    },
  };
}

function lbSpec() {
  return _sectionPlusCombinedSpec({
    sectionField: "lbPct",
    combinedField: "lbCombinedPct",
    sectionTitle: "Section LB (%)",
    combinedTitle: "Combined LB (%)",
    title: "Load balance (LB)",
    subtitle:
      "Solid: inference's own sections, LB = compute / compute_max. Dashed: combined per study, sum(compute) / sum(compute_max).",
  });
}

function commESpec() {
  return _sectionPlusCombinedSpec({
    sectionField: "commEPctSection",
    combinedField: "commEPct",
    sectionTitle: "Section CommE (%)",
    combinedTitle: "Combined CommE (%)",
    title: "Communication efficiency (CommE)",
    subtitle:
      "Solid: inference's own sections, CommE = compute_max / cuda_sec. Dashed: combined per study, sum(compute_max) / total_sec.",
  });
}

function combinedBandwidthSpec() {
  const panel = (yField: string, yTitle: string) => ({
    width: 420,
    height: 460,
    transform: [{ filter: "datum.gpuCount > 1 && datum.bandwidthGBs != null" }],
    mark: { type: "line", point: { size: 40 }, strokeWidth: 2.4 },
    encoding: {
      x: gpuAxis,
      y: { field: yField, type: "quantitative", title: yTitle, scale: { type: "log" } },
      color: {
        field: "section",
        type: "nominal",
        title: "Section",
        // Inference-only data (training's forward/backward aren't a clean
        // bandwidth read -- see the page text) -- an explicit domain
        // including sections absent from the data would still draw an
        // empty legend swatch for them.
        scale: {
          domain: INFERENCE.sections,
          range: INFERENCE.colors,
        },
      },
      strokeDash: dimDash,
      tooltip: [
        { field: "study", type: "nominal", title: "Study" },
        { field: "section", type: "nominal", title: "Section" },
        { field: "dim", type: "nominal", title: "Problem" },
        { field: "gpuCount", type: "quantitative", title: "GPUs" },
        { field: "nodeCount", type: "quantitative", title: "Nodes" },
        { field: yField, type: "quantitative", title: yTitle },
      ],
    },
  });

  return {
    $schema: "https://vega.github.io/schema/vega-lite/v6.json",
    description:
      "Inference's denoiser and gradient sections: transfer time and implied bandwidth vs GPU count.",
    title: {
      text: "Does the measured transfer time match real bandwidth?",
      subtitle:
        "Left: pure transfer time. Right: implied bandwidth -- the correctness check.",
    },
    config,
    hconcat: [
      panel("transferSec", "Transfer time (s, log)"),
      panel("bandwidthGBs", "Implied bandwidth (GB/s, log)"),
    ],
    resolve: { legend: { color: "shared" } },
  };
}

export const communicationShareInferenceSpec = shareSpec(INFERENCE);
export const communicationShareTrainingSpec = shareSpec(TRAINING);
export const communicationAbsoluteInferenceSpec = absoluteSpec(INFERENCE);
export const communicationAbsoluteTrainingSpec = absoluteSpec(TRAINING);
export const communicationWaitSpec = combinedWaitSpec();
export const communicationLbSpec = lbSpec();
export const communicationCommESpec = commESpec();
export const communicationBandwidthSpec = combinedBandwidthSpec();
