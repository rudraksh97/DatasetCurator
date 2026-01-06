/**
 * Badge component for status indicators and labels.
 */
import * as React from "react";
import clsx from "clsx";

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  /** Visual style variant */
  variant?: "default" | "secondary" | "success" | "warning" | "destructive";
}

/**
 * Inline badge component for displaying status or labels.
 */
export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = "default", ...props }, ref) => (
    <span
      ref={ref}
      className={clsx("badge", `badge-${variant}`, className)}
      {...props}
    />
  )
);
Badge.displayName = "Badge";
