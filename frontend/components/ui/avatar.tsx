/**
 * Avatar components for user/bot representation.
 */
import * as React from "react";
import clsx from "clsx";

export interface AvatarProps extends React.HTMLAttributes<HTMLDivElement> {}

/**
 * Avatar container component.
 */
export const Avatar = React.forwardRef<HTMLDivElement, AvatarProps>(
  ({ className, children, ...props }, ref) => (
    <div
      ref={ref}
      className={clsx("avatar", className)}
      {...props}
    >
      {children}
    </div>
  )
);
Avatar.displayName = "Avatar";

/**
 * Fallback content displayed when avatar image is unavailable.
 */
export const AvatarFallback = React.forwardRef<
  HTMLSpanElement,
  React.HTMLAttributes<HTMLSpanElement>
>(({ className, ...props }, ref) => (
  <span ref={ref} className={clsx("avatar-fallback", className)} {...props} />
));
AvatarFallback.displayName = "AvatarFallback";
