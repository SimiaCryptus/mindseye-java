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

@SuppressWarnings("serial")
public class AssertDimensionsLayer extends LayerBase {

  private final int[] dims;

  public AssertDimensionsLayer(final int... dims) {
    super();
    this.dims = dims;
  }

  protected AssertDimensionsLayer(@Nonnull final JsonObject json) {
    super(json);
    final JsonArray dimsJson = json.get("dims").getAsJsonArray();
    dims = IntStream.range(0, dimsJson.size()).map(i -> dimsJson.get(i).getAsInt()).toArray();
  }

  public static AssertDimensionsLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AssertDimensionsLayer(json);
  }

  @Override
  public Result evalAndFree(@Nonnull final Result... array) {
    if (0 == array.length) {
      throw new IllegalArgumentException(getName());
    }
    Result input = array[0];
    if (0 == input.getData().length()) {
      throw new IllegalArgumentException(getName());
    }
    @Nonnull final int[] inputDims = input.getData().getDimensions();
    if (Tensor.length(inputDims) != Tensor.length(dims)) {
      throw new IllegalArgumentException(getName() + ": " + Arrays.toString(inputDims) + " != " + Arrays.toString(dims));
    }
    return input;
  }

  @Override
  public List<Layer> getChildren() {
    return super.getChildren();
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    @Nonnull final JsonArray dimsJson = new JsonArray();
    for (final int dim : dims) {
      dimsJson.add(new JsonPrimitive(dim));
    }
    json.add("dims", dimsJson);
    return json;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

}
