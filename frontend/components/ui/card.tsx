/**
 * Card component for content containers.
 */
import React, { ReactNode, HTMLAttributes } from "react";
import clsx from "clsx";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Optional title displayed at the top of the card */
  title?: string;
  children?: ReactNode;
  className?: string;
}

/**
 * Container card component with optional title.
 */
export function Card({ title, children, className, ...props }: CardProps) {
  return (
    <div className={clsx("card", className)} {...props}>
      {title ? <h2 className="card-title">{title}</h2> : null}
      {children}
    </div>
  );
}
