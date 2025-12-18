import * as d3 from "https://esm.sh/d3@7.9.0";

function rgba_floats_tuple(colors, x) {
  if (x <= 0) {
    return colors[0];
  } else if (x >= 1) {
    return colors[colors.length - 1];
  }

  let i = Math.ceil(x * (colors.length - 1));
  let p = x - Math.floor(x);
  return d3
    .range(4)
    .map((j) => (1.0 - p) * colors[i - 1][j] + p * colors[i][j]);
}

function rgb_hex(colors, value) {
  let rgba_tuple = rgba_floats_tuple(colors, value).map((v) =>
    Math.floor(v * 255.9999),
  );
  let rgb_tuple = rgba_tuple.slice(0, 3);

  let hex = rgb_tuple.map((v) => v.toString(16).padStart(2, "0"));

  return `#${hex[0]}${hex[1]}${hex[2]}`;
}

function compute_value_domain(n_values, vmin, vmax) {
  return range(n_values).map(
    (index) => vmin + ((vmax - vmin) * index) / (n_values - 1),
  );
}

function subsample_colors(colors, n_colors) {
  return d3
    .range(n_colors)
    .map((i) => i / (n_colors - 1))
    .map((v) => rgb_hex(colors, v));
}

function draw_ramp(image, colors, width, height) {
  let border = image
    .append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", width)
    .attr("height", height)
    .style("stroke", "black")
    .style("stroke-width", 2);

  d3.range(width).forEach((index) => {
    let color = colors[index];

    return image
      .append("line")
      .attr("x1", index)
      .attr("y1", 0)
      .attr("x2", index)
      .attr("y2", height)
      .style("stroke", color)
      .style("stroke-width", 2);
  });

  return image;
}

function render({ model, el }) {
  let vmin = model.get("vmin");
  let vmax = model.get("vmax");

  const width = 640;
  const marginTop = 10;
  const marginRight = 20;
  const marginBottom = 40;
  const marginLeft = 40;
  const rampHeight = 40;
  const labelSpacer = 20;

  let widget_width = width - marginRight - marginLeft - 1;
  let colors = subsample_colors(model.get("colors"), widget_width);

  const x = d3
    .scaleLinear()
    .domain([vmin, vmax])
    .range([marginLeft, width - marginRight]);

  const svg = d3
    .select(el)
    .append("svg")
    .attr("width", width)
    .attr("height", 400);
  // axis label
  const label = svg
    .append("text")
    .attr("text-anchor", "middle")
    .attr("x", width / 2)
    .attr("y", marginTop + 5)
    .text(model.get("label"));
  const labelHeight = label.node().getBBox().height;

  // ramp
  let image = svg
    .append("g")
    .attr(
      "transform",
      `translate(${marginLeft + 1}, ${marginTop + labelHeight + labelSpacer})`,
    )
    .call((image) => draw_ramp(image, colors, widget_width, rampHeight));

  // ticks and tick labels
  const xaxis = svg
    .append("g")
    .attr(
      "transform",
      `translate(0,${marginTop + labelHeight + labelSpacer + rampHeight})`,
    )
    .call(d3.axisBottom(x));
  const axisHeight = xaxis.node().getBBox().height;

  // final styling
  const finalHeight =
    marginTop +
    labelHeight +
    labelSpacer +
    rampHeight +
    axisHeight +
    marginBottom;
  svg.attr("height", finalHeight);

  model.on("change:label", (model, new_label) => label.text(new_label));
  model.on("change:vmin", (model, new_vmin) => {
    let new_x = x.domain([new_vmin, model.get("vmax")]);
    xaxis.transition().duration(750).call(d3.axisBottom(new_x));
  });
  model.on("change:vmax", (model, new_vmax) => {
    let new_x = x.domain([model.get("vmin"), new_vmax]);
    xaxis.transition().duration(750).call(d3.axisBottom(new_x));
  });
  model.on("change:colors", (model, new_colors_array) => {
    let colors = subsample_colors(new_colors_array, widget_width);

    image
      .transition()
      .duration(250)
      .call((image) => {
        image.selectChildren().remove();
        draw_ramp(image.selection(), colors, widget_width, rampHeight);
      });
  });
}

export default { render };
