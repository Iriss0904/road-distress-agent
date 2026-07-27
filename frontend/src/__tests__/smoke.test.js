import { describe, expect, it } from "vitest";

import { formatMessage } from "../i18n/messages.js";

describe("public workbench shell", () => {
  it("renders a public-facing title without bundled knowledge data", () => {
    expect(formatMessage("en-US", "app.title")).toBe("Road Distress Treatment Workbench");
  });
});
