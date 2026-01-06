/**
 * ScrollArea component for scrollable content containers.
 */
import * as React from "react";
import clsx from "clsx";

export interface ScrollAreaProps extends React.HTMLAttributes<HTMLDivElement> {}

/**
 * Scrollable container with styled scrollbar.
 */
export const ScrollArea = React.forwardRef<HTMLDivElement, ScrollAreaProps>(
  ({ className, children, ...props }, ref) => (
    <div
      ref={ref}
      className={clsx("scroll-area", className)}
      {...props}
    >
      {children}
    </div>
  )
);
ScrollArea.displayName = "ScrollArea";
