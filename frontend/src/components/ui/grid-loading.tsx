"use client"

import { cn } from "@/lib/utils"

interface GridLoadingProps {
  size?: "sm" | "md" | "lg"
  className?: string
}

export default function GridLoading({
  size = "md",
  className,
}: GridLoadingProps) {
  const containerSizes = {
    sm: "w-8 h-8",
    md: "w-12 h-12",
    lg: "w-16 h-16",
  }

  const gapSizes = {
    sm: "gap-[2px]",
    md: "gap-[3px]",
    lg: "gap-1",
  }

  return (
    <div className={cn("relative", containerSizes[size], className)}>
      <div className={cn("grid grid-cols-3 w-full h-full", gapSizes[size])}>
        {Array.from({ length: 9 }).map((_, i) => (
          <div
            key={i}
            className="rounded-[2px] grid-loading-cell"
            style={{
              animationDelay: `${i * 0.1}s`,
            }}
          />
        ))}
      </div>
    </div>
  )
}
