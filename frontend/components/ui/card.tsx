import React, { ReactNode } from "react";
import clsx from "clsx";

interface CardProps {
  title?: string;
  children?: ReactNode;
  className?: string;
}

export function Card({ title, children, className }: CardProps) {
  return (
    <div className={clsx("card", className)}>
      {title ? <h2 className="card-title">{title}</h2> : null}
      {children}
    </div>
  );
}

