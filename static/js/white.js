window.White = (() => {
  const palette = ["#2563eb", "#16a34a", "#f97316", "#dc2626", "#7c3aed", "#0891b2"];

  function drawBarChart(canvas, rows) {
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const padding = 48;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 1.7;
    ctx.clearRect(0, 0, width, height);
    ctx.font = "14px Inter, Arial, sans-serif";
    ctx.fillStyle = "#0f172a";
    ctx.fillText("Model accuracy (%)", padding, 26);

    const maxValue = 100;
    const barWidth = chartWidth / Math.max(rows.length, 1) - 18;
    rows.forEach((row, index) => {
      const value = Math.round((row.accuracy || 0) * 1000) / 10;
      const x = padding + index * (chartWidth / rows.length) + 8;
      const barHeight = (value / maxValue) * chartHeight;
      const y = height - padding - barHeight;
      ctx.fillStyle = palette[index % palette.length];
      ctx.fillRect(x, y, Math.max(barWidth, 20), barHeight);
      ctx.fillStyle = "#334155";
      ctx.fillText(`${value}%`, x, y - 8);
      ctx.save();
      ctx.translate(x + 4, height - 18);
      ctx.rotate(-0.32);
      ctx.fillText(row.model, 0, 0);
      ctx.restore();
    });

    ctx.strokeStyle = "#cbd5e1";
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();
  }

  function drawPredictionChart(canvas, rows) {
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const padding = 46;
    const chartHeight = height - padding * 2;
    const total = rows.reduce((sum, row) => sum + Number(row.value || 0), 0);
    ctx.clearRect(0, 0, width, height);
    ctx.font = "14px Inter, Arial, sans-serif";
    ctx.fillStyle = "#0f172a";
    ctx.fillText("Fraud vs legitimate predictions", padding, 26);

    if (!total) {
      ctx.fillStyle = "#64748b";
      ctx.fillText("No prediction logs yet", padding, height / 2);
      return;
    }

    const barWidth = (width - padding * 2) / rows.length - 28;
    rows.forEach((row, index) => {
      const value = Number(row.value || 0);
      const percent = Math.round((value / total) * 1000) / 10;
      const x = padding + index * ((width - padding * 2) / rows.length) + 14;
      const barHeight = (value / total) * chartHeight;
      const y = height - padding - barHeight;
      ctx.fillStyle = row.label === "Fraud" ? "#dc2626" : "#16a34a";
      ctx.fillRect(x, y, Math.max(barWidth, 40), barHeight);
      ctx.fillStyle = "#334155";
      ctx.fillText(`${row.label}: ${value}`, x, height - 20);
      ctx.fillText(`${percent}%`, x, y - 8);
    });

    ctx.strokeStyle = "#cbd5e1";
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();
  }

  function boot() {
    document.querySelectorAll(".white-chart").forEach((canvas) => {
      const rows = JSON.parse(canvas.dataset.chart || "[]");
      if (canvas.dataset.chartType === "prediction") {
        drawPredictionChart(canvas, rows);
      } else {
        drawBarChart(canvas, rows);
      }
    });
  }

  return { boot, drawBarChart, drawPredictionChart };
})();
