import { NextResponse } from "next/server";

// Static-Export-kompatibel: Route wird zur Build-Zeit gerendert (APK inklusive).
export const dynamic = "force-static";

export async function GET() {
  return NextResponse.json({
    app: "MicroDuck Trainer",
    version: "2.0",
    ok: true,
  });
}
