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

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.*;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.ref.wrappers.RefIntStream;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware class AssertDimensionsLayer extends LayerBase {

  private final int[] dims;

  public AssertDimensionsLayer(final int... dims) {
    super();
    this.dims = dims;
  }

  protected AssertDimensionsLayer(@Nonnull final JsonObject json) {
    super(json);
    final JsonArray dimsJson = json.get("dims").getAsJsonArray();
    dims = com.simiacryptus.ref.wrappers.RefIntStream.range(0, dimsJson.size()).map(i -> dimsJson.get(i).getAsInt())
        .toArray();
  }

  @Override
  public com.simiacryptus.ref.wrappers.RefList<Layer> getChildren() {
    return super.getChildren();
  }

  @SuppressWarnings("unused")
  public static AssertDimensionsLayer fromJson(@Nonnull final JsonObject json,
      com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new AssertDimensionsLayer(json);
  }

  @Override
  public Result eval(@Nonnull final Result... array) {
    if (0 == array.length) {
      throw new IllegalArgumentException(getName());
    }
    Result input = array[0];
    if (0 == input.getData().length()) {
      throw new IllegalArgumentException(getName());
    }
    @Nonnull
    final int[] inputDims = input.getData().getDimensions();
    if (Tensor.length(inputDims) != Tensor.length(dims)) {
      throw new IllegalArgumentException(getName() + ": " + com.simiacryptus.ref.wrappers.RefArrays.toString(inputDims)
          + " != " + com.simiacryptus.ref.wrappers.RefArrays.toString(dims));
    }
    return input;
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
      DataSerializer dataSerializer) {
    @Nonnull
    final JsonObject json = super.getJsonStub();
    @Nonnull
    final JsonArray dimsJson = new JsonArray();
    for (final int dim : dims) {
      dimsJson.add(new JsonPrimitive(dim));
    }
    json.add("dims", dimsJson);
    return json;
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
  }

  public @SuppressWarnings("unused") void _free() {
  }

  public @Override @SuppressWarnings("unused") AssertDimensionsLayer addRef() {
    return (AssertDimensionsLayer) super.addRef();
  }

  public static @SuppressWarnings("unused") AssertDimensionsLayer[] addRefs(AssertDimensionsLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(AssertDimensionsLayer::addRef)
        .toArray((x) -> new AssertDimensionsLayer[x]);
  }

  public static @SuppressWarnings("unused") AssertDimensionsLayer[][] addRefs(AssertDimensionsLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(AssertDimensionsLayer::addRefs)
        .toArray((x) -> new AssertDimensionsLayer[x][]);
  }

}
