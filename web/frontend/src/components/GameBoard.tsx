import { useRef, useEffect, useCallback } from "react";
import type { GameFrame } from "../types/game";
import { effectiveCellSize, detailLevel } from "../lib/viewport";
import { drawGrid, drawTerrainMap, drawFreeSlots, drawFruits, drawTargetArrows, drawAgents } from "../lib/renderer";

interface Props {
  frame: GameFrame | null;
}

export default function GameBoard({ frame }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef(1);
  const panRef = useRef({ x: 0, y: 0 });
  const dragRef = useRef<{ startX: number; startY: number; panX: number; panY: number } | null>(null);
  const imagesRef = useRef<{ agent: HTMLImageElement | null; apple: HTMLImageElement | null }>({
    agent: null,
    apple: null,
  });
  const sizeRef = useRef({ w: 600, h: 600 });

  // Preload sprites
  useEffect(() => {
    const agentImg = new Image();
    agentImg.src = "/agent.png";
    agentImg.onload = () => {
      imagesRef.current.agent = agentImg;
    };
    const appleImg = new Image();
    appleImg.src = "/apple.png";
    appleImg.onload = () => {
      imagesRef.current.apple = appleImg;
    };
  }, []);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { w, h } = sizeRef.current;
    canvas.width = w * window.devicePixelRatio;
    canvas.height = h * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    ctx.clearRect(0, 0, w, h);

    const canvasPx = Math.min(w, h);
    const cell = effectiveCellSize(canvasPx, frame.field_size, zoomRef.current);
    const dl = detailLevel(cell);
    const { x: panX, y: panY } = panRef.current;

    drawGrid(ctx, frame.field_size, cell, panX, panY);
    drawTerrainMap(ctx, frame.ca_map ?? null, frame.food_growth ?? {}, frame.field_size, cell, panX, panY);
    drawFreeSlots(ctx, frame.fruits, cell, panX, panY, dl);
    drawFruits(ctx, frame.fruits, cell, panX, panY, imagesRef.current.apple, dl);
    drawTargetArrows(ctx, frame.agents, cell, panX, panY);
    drawAgents(ctx, frame.agents, cell, panX, panY, imagesRef.current.agent, dl);
  }, [frame]);

  // Redraw on frame change
  useEffect(() => {
    requestAnimationFrame(redraw);
  }, [redraw]);

  // ResizeObserver for responsive canvas
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        sizeRef.current = {
          w: entry.contentRect.width,
          h: entry.contentRect.height,
        };
        requestAnimationFrame(redraw);
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [redraw]);

  // Zoom
  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const canvas = canvasRef.current;
      if (!canvas || !frame) return;

      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;

      const canvasPx = Math.min(sizeRef.current.w, sizeRef.current.h);
      const oldCell = effectiveCellSize(canvasPx, frame.field_size, zoomRef.current);
      const worldCol = (cx - panRef.current.x) / oldCell;
      const worldRow = (cy - panRef.current.y) / oldCell;

      const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
      zoomRef.current = Math.max(0.5, Math.min(10, zoomRef.current * factor));

      const newCell = effectiveCellSize(canvasPx, frame.field_size, zoomRef.current);
      panRef.current.x = cx - worldCol * newCell;
      panRef.current.y = cy - worldRow * newCell;

      requestAnimationFrame(redraw);
    },
    [frame, redraw]
  );

  // Pan
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      panX: panRef.current.x,
      panY: panRef.current.y,
    };
  }, []);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragRef.current) return;
      panRef.current.x = dragRef.current.panX + (e.clientX - dragRef.current.startX);
      panRef.current.y = dragRef.current.panY + (e.clientY - dragRef.current.startY);
      requestAnimationFrame(redraw);
    },
    [redraw]
  );

  const handleMouseUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  return (
    <div ref={containerRef} className="game-board">
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", cursor: dragRef.current ? "grabbing" : "grab" }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
    </div>
  );
}
