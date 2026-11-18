const margins = { top: 40, right: 30, bottom: 60, left: 60};
const width = 800 - margins.left - margins.right;
const height = 650 - margins.top - margins.bottom;

const sheldus_loss_svg = d3.select("#sheldus_loss")
.append("svg")
.attr("width", width + margins.left + margins.right)
.attr("height", height + margins.bottom + margins.bottom)
.append("g")
.attr("transform", `translate(${margins.left}, ${margins.top})`);

const tornado_days_svg = d3.select("#tornado_days").append("svg")
.append("svg")
.attr("width", width + margins.left + margins.right)
.attr("height", height + margins.bottom + margins.bottom)
.append("g")
.attr("transform", `translate(${margins.left}, ${margins.top})`);

var g = sheldus_loss_svg.append("g");

const imageData = [
    { x: 0, y: 0, url: "output/graph_sheldus_loss.png", width: 400, height: 600 },
    { x: 400, y: 150, url: "output/graph_tornado_days.png", width: 300, height: 400 },
    { x: 350, y: 600, url: "output/qa_series_sheldus_loss.png", width: 120, height: 90 }
];


function generate_data(){
    sheldus_loss_svg.selectAll("image")
    .data(imageData)
    .enter()
    .append("image")
    .attr("x", d => d.x)
    .attr("y", d => d.y)
    .attr("width", d => d.width)
    .attr("height", d => d.height)
    .attr("href", d => d.url);
}