const colors = ["#168d80", "#f2b46d", "#50617d"];

const config = {
  background: null,
  font: "Inter, ui-sans-serif, system-ui, sans-serif",
  view: { stroke: null },
  axis: {
    domainColor: "#8290a8",
    gridColor: "#e5e8e2",
    labelColor: "#50617d",
    titleColor: "#20304e",
    titleFontWeight: 600,
  },
  legend: {
    labelColor: "#50617d",
    titleColor: "#20304e",
    orient: "bottom",
    direction: "horizontal",
  },
};

const denoiserColor = {
  field: "denoiser",
  type: "nominal",
  title: "Denoiser",
  scale: { domain: ["dncnn", "drunet", "unet"], range: colors },
};

const dimShape = {
  field: "dim",
  type: "nominal",
  title: "Dimensionality",
  scale: { domain: [2, 3], range: ["circle", "square"] },
  legend: { labelExpr: "datum.label === '2' ? '2D' : '3D'" },
};

export const denoiserSpeedupSpec = {
  $schema: "https://vega.github.io/schema/vega-lite/v6.json",
  description: "torch.compile steady-state speedup per denoiser, shape, and GPU.",
  title: {
    text: "torch.compile speedup (eager / compiled)",
    subtitle: "Steady-state per-iteration denoiser time, compile=None vs compile=pre.",
  },
  config,
  facet: {
    column: {
      field: "gpu",
      type: "nominal",
      title: "GPU",
      header: { labelFontSize: 12, titleFontSize: 12 },
    },
  },
  spec: {
    width: 380,
    height: 440,
    layer: [
      {
        mark: { type: "rule", strokeDash: [6, 5], color: "#8290a8" },
        encoding: { y: { datum: 1 } },
      },
      {
        mark: { type: "line", point: { filled: true, size: 70 }, strokeWidth: 2 },
        encoding: {
          x: {
            field: "shape",
            type: "ordinal",
            title: "Shape (2D | 3D)",
            sort: null,
          },
          y: {
            field: "speedup",
            type: "quantitative",
            title: "Speedup",
            scale: { zero: false },
          },
          color: denoiserColor,
          shape: dimShape,
          detail: [{ field: "denoiser" }, { field: "dim" }],
          tooltip: [
            { field: "gpu", type: "nominal", title: "GPU" },
            { field: "denoiser", type: "nominal", title: "Denoiser" },
            { field: "shape", type: "ordinal", title: "Shape" },
            { field: "eagerSec", type: "quantitative", title: "Eager (s)", format: ".4f" },
            {
              field: "compiledSec",
              type: "quantitative",
              title: "Compiled (s)",
              format: ".4f",
            },
            { field: "speedup", type: "quantitative", title: "Speedup", format: ".2f" },
          ],
        },
      },
    ],
  },
  resolve: { scale: { x: "independent" } },
};

export const denoiserRooflineSpec = (ridgePointFlopByte: number) => ({
  $schema: "https://vega.github.io/schema/vega-lite/v6.json",
  description: "Arithmetic intensity vs torch.compile speedup on H100.",
  title: {
    text: "Roofline: arithmetic intensity vs compile speedup — H100",
    subtitle: "Points left of the ridge are memory-bound, right are compute-bound.",
  },
  width: 620,
  height: 380,
  config,
  layer: [
    {
      mark: { type: "rule", strokeDash: [6, 5], color: "#8290a8" },
      encoding: { y: { datum: 1 } },
    },
    {
      mark: { type: "rule", strokeDash: [6, 5], color: "#8290a8" },
      encoding: { x: { datum: ridgePointFlopByte, type: "quantitative" } },
    },
    {
      mark: {
        type: "point",
        filled: true,
        size: 170,
        stroke: "#20304e",
        strokeWidth: 0.8,
      },
      encoding: {
        x: {
          field: "arithIntensityFlopByte",
          type: "quantitative",
          title: "Arithmetic intensity (FLOP / byte)",
          scale: { type: "log" },
        },
        y: {
          field: "speedup",
          type: "quantitative",
          title: "Speedup (eager / compiled)",
        },
        color: denoiserColor,
        shape: dimShape,
        tooltip: [
          { field: "denoiser", type: "nominal", title: "Denoiser" },
          { field: "shape", type: "ordinal", title: "Shape" },
          {
            field: "arithIntensityFlopByte",
            type: "quantitative",
            title: "FLOP/byte",
            format: ".1f",
          },
          { field: "speedup", type: "quantitative", title: "Speedup", format: ".2f" },
        ],
      },
    },
  ],
});

const stageColor = {
  field: "stage",
  type: "nominal",
  title: null,
  scale: {
    domain: ["firstIterSec", "stableIterSec"],
    range: [colors[1], colors[0]],
  },
  legend: {
    labelExpr: "datum.label === 'firstIterSec' ? '1st iteration' : 'stable'",
  },
};

export const pnpCompileModesSpec = {
  $schema: "https://vega.github.io/schema/vega-lite/v6.json",
  description: "PnP 1st-iteration vs stable-iteration cost across compile modes.",
  title: {
    text: "torch.compile: 1st-iteration vs stable-iteration cost",
    subtitle: "PnP reconstruction, compile=None/pre/post/fused, 1 vs 2 GPU.",
  },
  config,
  facet: {
    column: {
      field: "gpuCount",
      type: "ordinal",
      title: "GPU count",
      header: { labelFontSize: 12, titleFontSize: 12 },
    },
  },
  spec: {
    width: 320,
    height: 440,
    transform: [
      { fold: ["firstIterSec", "stableIterSec"], as: ["stage", "seconds"] },
    ],
    mark: { type: "bar" },
    encoding: {
      x: {
        field: "compileMode",
        type: "nominal",
        title: "Compile mode",
        sort: ["None", "pre", "post", "fused"],
      },
      xOffset: { field: "stage", sort: ["firstIterSec", "stableIterSec"] },
      y: {
        field: "seconds",
        type: "quantitative",
        title: "Time / iteration (s, log scale)",
        scale: { type: "log", domain: [0.2, 30] },
      },
      y2: { datum: 0.2 },
      color: stageColor,
      tooltip: [
        { field: "compileMode", type: "nominal", title: "Compile mode" },
        { field: "gpuCount", type: "quantitative", title: "GPUs" },
        { field: "stage", type: "nominal", title: "Stage" },
        { field: "seconds", type: "quantitative", title: "Seconds", format: ".4f" },
        { field: "speedup", type: "quantitative", title: "Speedup", format: ".2f" },
      ],
    },
  },
};
