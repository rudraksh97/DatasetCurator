/**
 * Table components for data display.
 */
import { HTMLAttributes } from "react";
import clsx from "clsx";

/** Table wrapper component */
export function Table({ className, ...props }: HTMLAttributes<HTMLTableElement>) {
  return <table className={clsx("table", className)} {...props} />;
}

/** Table header section */
export function THead({ className, ...props }: HTMLAttributes<HTMLTableSectionElement>) {
  return <thead className={className} {...props} />;
}

/** Table body section */
export function TBody({ className, ...props }: HTMLAttributes<HTMLTableSectionElement>) {
  return <tbody className={className} {...props} />;
}

/** Table row */
export function TR({ className, ...props }: HTMLAttributes<HTMLTableRowElement>) {
  return <tr className={className} {...props} />;
}

/** Table header cell */
export function TH({ className, ...props }: HTMLAttributes<HTMLTableCellElement>) {
  return <th className={className} {...props} />;
}

/** Table data cell */
export function TD({ className, ...props }: HTMLAttributes<HTMLTableCellElement>) {
  return <td className={className} {...props} />;
}
