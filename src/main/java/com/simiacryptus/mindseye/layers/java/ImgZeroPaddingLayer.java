/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import java.util.Map;

/**
 * The type Img zero padding layer.
 */
@SuppressWarnings("serial")
public class ImgZeroPaddingLayer extends LayerBase {

  private final int sizeX;
  private final int sizeY;

  /**
   * Instantiates a new Img zero padding layer.
   *
   * @param sizeX the size x
   * @param sizeY the size y
   */
  public ImgZeroPaddingLayer(final int sizeX, final int sizeY) {
    super();
    this.sizeX = sizeX;
    this.sizeY = sizeY;
  }

  /**
   * Instantiates a new Img zero padding layer.
   *
   * @param json the json
   */
  protected ImgZeroPaddingLayer(@Nonnull final JsonObject json) {
    super(json);
    sizeX = json.getAsJsonPrimitive("sizeX").getAsInt();
    sizeY = json.getAsJsonPrimitive("sizeY").getAsInt();
  }

  /**
   * From json img zero padding layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img zero padding layer
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static ImgZeroPaddingLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgZeroPaddingLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert inObj.length == 1;
    TensorList tensorList = inObj[0].getData();
    @Nonnull
    int[] dimensions = tensorList.getDimensions();
    tensorList.freeRef();
    ImgCropLayer imgCropLayer = new ImgCropLayer(dimensions[0] + 2 * this.sizeX, dimensions[1] + 2 * this.sizeY);
    Result result = imgCropLayer.eval(inObj);
    imgCropLayer.freeRef();
    return result;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.addProperty("sizeX", sizeX);
    json.addProperty("sizeY", sizeX);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return new RefArrayList<>();
  }

  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  ImgZeroPaddingLayer addRef() {
    return (ImgZeroPaddingLayer) super.addRef();
  }

}
