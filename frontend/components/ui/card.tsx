/**
 * Card component for content containers.
 */
import React, { ReactNode } from "react";
import clsx from "clsx";

interface CardProps {
  /** Optional title displayed at the top of the card */
  title?: string;
  children?: ReactNode;
  className?: string;
}

/**
 * Container card component with optional title.
 */
export function Card({ title, children, className }: CardProps) {
  return (
    <div className={clsx("card", className)}>
      {title ? <h2 className="card-title">{title}</h2> : null}
      {children}
    </div>
  );
}
