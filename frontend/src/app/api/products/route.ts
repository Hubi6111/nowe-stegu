import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const DEFAULT_SHOP_URL = "https://stegu.pl/produkty/plytki-ceglopodobne/";

interface ProductMetadata {
  name: string;
  moduleWidthMm: number;
  moduleHeightMm: number;
  jointMm: number;
  layoutType: string;
  offsetRatio: number;
  shopUrl?: string;
}

interface Product extends ProductMetadata {
  productId: string;
  textureImage: string;
  shopUrl: string;
}

const REQUIRED_FIELDS: (keyof ProductMetadata)[] = [
  "name",
  "moduleWidthMm",
  "moduleHeightMm",
  "jointMm",
  "layoutType",
  "offsetRatio",
];

function getTexturesDir(): string | null {
  const candidates = [
    path.join(process.cwd(), "assets", "textures", "stegu"),
    path.join(process.cwd(), "..", "assets", "textures", "stegu"),
    path.join(process.cwd(), "frontend", "assets", "textures", "stegu"),
  ];
  for (const dir of candidates) {
    if (fs.existsSync(dir)) return dir;
  }
  return null;
}

export async function GET() {
  const texturesDir = getTexturesDir();

  if (!texturesDir) {
    return NextResponse.json([], { status: 200 });
  }

  const entries = fs.readdirSync(texturesDir, { withFileTypes: true });
  const folders = entries
    .filter((e) => e.isDirectory())
    .sort((a, b) => a.name.localeCompare(b.name));

  if (folders.length === 0) {
    return NextResponse.json([]);
  }

  const products: Product[] = [];
  const errors: string[] = [];

  for (const folder of folders) {
    const folderPath = path.join(texturesDir, folder.name);
    const metadataPath = path.join(folderPath, "metadata.json");
    const albedoPath = path.join(folderPath, "albedo.jpg");

    if (!fs.existsSync(metadataPath)) {
      errors.push(`metadata.json missing in ${folder.name}`);
      continue;
    }
    if (!fs.existsSync(albedoPath)) {
      errors.push(`albedo.jpg missing in ${folder.name}`);
      continue;
    }

    try {
      const raw = JSON.parse(fs.readFileSync(metadataPath, "utf-8"));
      const missing = REQUIRED_FIELDS.filter((f) => !(f in raw));
      if (missing.length > 0) {
        errors.push(
          `metadata.json in ${folder.name} is missing: ${missing.join(", ")}`
        );
        continue;
      }

      products.push({
        productId: folder.name,
        name: raw.name,
        textureImage: "albedo.jpg",
        moduleWidthMm: raw.moduleWidthMm,
        moduleHeightMm: raw.moduleHeightMm,
        jointMm: raw.jointMm,
        layoutType: raw.layoutType,
        offsetRatio: raw.offsetRatio,
        shopUrl: raw.shopUrl || DEFAULT_SHOP_URL,
      });
    } catch {
      errors.push(`Invalid JSON in ${folder.name}/metadata.json`);
    }
  }

  if (errors.length > 0) {
    return NextResponse.json(
      { message: "Some product folders have problems", errors },
      { status: 500 }
    );
  }

  return NextResponse.json(products);
}
